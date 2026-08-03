import pytest
import torch

from src.quant.constants import _INT8_FORMATS
from src.quant.quantize import (
    quantize_operand,
    dequantize_operand,
    effective_block_size,
)
from src.quant.utils import str_to_qmax


def _fp8_kw(scale_dtype=None):
    return dict(fmt="fp8_e4m3", scale_dtype=scale_dtype)


def _int_kw(fmt="int8"):
    return dict(fmt=fmt)


def _roundtrip(x, contract_dim, granularity, block_size, fmt, *, scale_dtype=None):
    """dequant(quant(x)) — the reconstruction the quantized GEMMs actually see."""
    xq, scale = quantize_operand(
        x, contract_dim, granularity, block_size, fmt, scale_dtype=scale_dtype
    )
    return dequantize_operand(xq, scale, contract_dim, granularity, block_size)


# --- scale block sizing ---


def test_effective_block_size():
    assert effective_block_size("blockwise", 32, 256) == 32
    assert effective_block_size("rowwise", 128, 256) == 256
    assert effective_block_size("tensorwise", 128, 256) == 256


# --- operand quantize: canonical scale shapes ---


@pytest.mark.parametrize(
    "gran,bs,nkb", [("tensorwise", 256, 1), ("rowwise", 256, 1), ("blockwise", 64, 4)]
)
def test_operand_a_scale_shape(gran, bs, nkb):
    x = torch.randn(8, 256)
    xq, s = quantize_operand(x, -1, gran, bs, **_fp8_kw())
    assert xq.shape == (8, 256) and xq.dtype == torch.float8_e4m3fn
    assert s.shape == (8, nkb) and s.dtype == torch.float32 and s.is_contiguous()


@pytest.mark.parametrize(
    "gran,bs,nkb", [("tensorwise", 256, 1), ("rowwise", 256, 1), ("blockwise", 64, 4)]
)
def test_operand_b_scale_shape(gran, bs, nkb):
    x = torch.randn(256, 12)
    xq, s = quantize_operand(x, 0, gran, bs, **_fp8_kw())
    assert xq.shape == (256, 12)
    assert s.shape == (nkb, 12) and s.is_contiguous()


def test_blockwise_partial_last_block():
    # K=100 not a multiple of 32 -> nkb=4, last block covers [96,100)
    x = torch.randn(4, 100)
    xq, s = quantize_operand(x, -1, "blockwise", 32, **_fp8_kw())
    assert xq.shape == (4, 100) and s.shape == (4, 4)


# --- roundtrip error, input dtypes, zero-safety ---


def test_rowwise_fp8_roundtrips_within_error():
    torch.manual_seed(0)
    x = torch.randn(16, 128)
    fq = _roundtrip(x, -1, "rowwise", 128, **_fp8_kw())
    assert (fq - x).norm() / x.norm() < 0.1


def test_tensorwise_fp8_roundtrips_within_error():
    torch.manual_seed(0)
    x = torch.randn(64, 64) * 10.0
    xq, _ = quantize_operand(x, -1, "tensorwise", 0, **_fp8_kw())
    assert xq.dtype == torch.float8_e4m3fn
    recon = _roundtrip(x, -1, "tensorwise", 0, **_fp8_kw())
    assert (recon - x).norm() / x.norm() < 0.1  # e4m3 per-tensor dynamic range


@pytest.mark.parametrize("gran,bs", [("tensorwise", 0), ("blockwise", 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_quantize_operand_accepts_all_compute_dtypes(gran, bs, dtype):
    x = torch.randn(8, 64, dtype=dtype)
    xq, s = quantize_operand(x, -1, gran, bs, **_fp8_kw())
    assert xq.shape == x.shape and xq.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.float32  # scale is fp32 regardless of input dtype


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
def test_zero_input_safe(gran, bs):
    x = torch.zeros(4, 64)
    fq = _roundtrip(x, -1, gran, bs, **_fp8_kw())
    assert torch.isfinite(fq).all()


# --- int operand quantize ---


def test_int_quant_rounds_and_clamps():
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 10
    xq, s = quantize_operand(x, -1, "rowwise", 64, **_int_kw("int4"))
    assert xq.dtype == torch.int8
    assert int(xq.min()) >= -7 and int(xq.max()) <= 7


@pytest.mark.parametrize("fmt", _INT8_FORMATS)
def test_int_tensorwise_symmetric_range_and_scale(fmt):
    torch.manual_seed(0)
    qmax = str_to_qmax(fmt)
    x = torch.randn(64, 64) * 10.0
    q, s = quantize_operand(x, -1, "tensorwise", 0, **_int_kw(fmt))
    assert q.dtype == torch.int8  # all widths stored in int8
    assert int(q.min()) >= -qmax and int(q.max()) <= qmax
    # tensorwise scale broadcasts a single value = amax / qmax over the rows
    assert torch.allclose(s, x.abs().max().float() / qmax)


@pytest.mark.parametrize("fmt", _INT8_FORMATS)
def test_int_tensorwise_recon_within_half_scale(fmt):
    torch.manual_seed(0)
    qmax = str_to_qmax(fmt)
    x = torch.randn(128, 128)
    recon = _roundtrip(x, -1, "tensorwise", 0, **_int_kw(fmt))
    scale = x.abs().max().float() / qmax
    assert (recon - x).abs().max() <= scale.item() / 2 + 1e-6


# --- power-of-two (E8M0 / MX-style) scale ---


def test_power_of_two_scale_is_exact_pow2():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    _, s = quantize_operand(x, -1, "blockwise", 32, **_fp8_kw(scale_dtype="fp8_e8m0"))
    log2s = torch.log2(s[s > 0])
    assert torch.allclose(log2s, log2s.round())


def test_pow2_scale_block32_shape_and_exactness():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    xq, s = quantize_operand(x, -1, "blockwise", 32, **_fp8_kw(scale_dtype="fp8_e8m0"))
    assert xq.dtype == torch.float8_e4m3fn and s.shape == (4, 2)
    log2s = torch.log2(s[s > 0])
    assert torch.allclose(log2s, log2s.round())


def test_pow2_scale_roundtrips_within_error():
    torch.manual_seed(0)
    x = torch.randn(8, 128)
    fq = _roundtrip(x, -1, "blockwise", 32, **_fp8_kw(scale_dtype="fp8_e8m0"))
    assert (fq - x).norm() / x.norm() < 0.15


@pytest.mark.parametrize("granularity", ["tensorwise", "rowwise"])
def test_fp32_scale_never_clips(granularity):
    # scale = amax/qmax lands the extremal element exactly on qmax, so the clamp in
    # quantize_operand is a no-op and the largest magnitude round-trips intact.
    torch.manual_seed(0)
    x = torch.randn(32, 64) * 10.0
    fq = _roundtrip(x, -1, granularity, 0, **_fp8_kw())
    assert fq.abs().amax().item() == pytest.approx(x.abs().amax().item(), rel=1e-6)


def test_e8m0_scale_clips_the_block_maximum():
    # e8m0 floors the exponent, so the scale is up to 2x too small and codes above
    # qmax saturate: 1.9 -> floor(log2 1.9) = 0 -> scale = 2^-8 -> 1.9/scale = 486 > 448,
    # clamped back to 448 * 2^-8 = 1.75.
    x = torch.full((1, 32), 1.9)
    xq, s = quantize_operand(x, -1, "blockwise", 32, **_fp8_kw(scale_dtype="fp8_e8m0"))
    assert xq.float().abs().amax().item() == str_to_qmax(
        "fp8_e4m3"
    )  # every code clipped
    fq = dequantize_operand(xq, s, -1, "blockwise", 32)
    assert torch.allclose(fq, torch.full_like(fq, 1.75))


def test_e8m0_clipping_loss_is_bounded_by_one_eighth():
    # Worst case is a mantissa just under 2: qmax/2^emax = 1.75 of it survives, so no
    # element loses more than (2 - 1.75) / 2 = 12.5% of its magnitude.
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 3.0
    fq = _roundtrip(x, -1, "blockwise", 32, **_fp8_kw(scale_dtype="fp8_e8m0"))
    clipped = x.abs() > fq.abs()
    loss = (x.abs() - fq.abs())[clipped] / x.abs()[clipped]
    assert loss.max().item() <= 0.125


# --- dequantize_operand ---


def _manual_dequant(x, contract_dim, bs, *, fmt, scale_dtype=None):
    """Reference dequant using the *true* block size `bs` (never inferred from
    `nkb`/`K`), to regression-test that dequantize_operand's own bs recovery
    matches it even when the last block is partial (K not a multiple of bs)."""
    xq, s = quantize_operand(
        x, contract_dim, "blockwise", bs, fmt, scale_dtype=scale_dtype
    )
    qf = xq.movedim(contract_dim, -1).float()
    K = qf.shape[-1]
    sf = s.movedim(contract_dim, -1).float()
    nkb = sf.shape[-1]
    pad = nkb * bs - K
    qp = torch.nn.functional.pad(qf, (0, pad)) if pad else qf
    deq = (qp.unflatten(-1, (nkb, bs)) * sf.unsqueeze(-1)).flatten(-2)[..., :K]
    return deq.movedim(-1, contract_dim)


@pytest.mark.parametrize("contract_dim", [-1, 0])
@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
def test_dequantize_operand_preserves_shape_and_layout(gran, bs, contract_dim):
    torch.manual_seed(0)
    x = torch.randn(64, 96)
    xq, s = quantize_operand(x, contract_dim, gran, bs, **_fp8_kw())
    deq = dequantize_operand(xq, s, contract_dim, gran, bs)
    assert deq.shape == x.shape and deq.is_contiguous()


def test_dequantize_operand_returns_float32_from_bf16():
    # metrics compare in fp32 regardless of the operand's compute dtype
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    xq, s = quantize_operand(x, -1, "rowwise", 0, **_fp8_kw())
    assert dequantize_operand(xq, s, -1, "rowwise", 0).dtype == torch.float32


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
def test_dequantize_operand_stacked_experts_matches_per_expert(gran, bs):
    # The MoE path quantizes each (K,N) expert slab along dim 0 and stacks, then
    # inverts the (E,K,N) result in one call at contract_dim=1. Assert that agrees
    # with dequantizing slab by slab — the two paths are written differently.
    torch.manual_seed(0)
    experts = torch.randn(4, 96, 32)
    per_expert = [
        quantize_operand(experts[g], 0, gran, bs, **_fp8_kw())
        for g in range(experts.shape[0])
    ]
    deq = dequantize_operand(
        torch.stack([q for q, _ in per_expert]),
        torch.stack([s for _, s in per_expert]),
        1,
        gran,
        bs,
    )
    reference = torch.stack(
        [
            _roundtrip(experts[g], 0, gran, bs, **_fp8_kw()).float()
            for g in range(experts.shape[0])
        ]
    )
    assert torch.equal(deq, reference)


def test_dequantize_operand_partial_block_uses_given_block_size():
    # K=100, bs=32 -> nkb=4; an inferred ceil(100/4)=25 would diverge.
    torch.manual_seed(0)
    x = torch.randn(4, 100)
    xq, s = quantize_operand(x, -1, "blockwise", 32, **_fp8_kw())
    deq = dequantize_operand(xq, s, -1, "blockwise", 32)
    assert torch.allclose(deq, _manual_dequant(x, -1, 32, **_fp8_kw()), atol=1e-6)
