import pytest
import torch

from src.quant.constants import _INT8_FORMATS, _STR_TO_DTYPE, _STR_TO_QMAX
from src.quant.quantize import (
    quantize_operand,
    fake_quantize_operand,
    effective_block_size,
)

E4M3 = _STR_TO_DTYPE["fp8_e4m3"]

# qmax per int fmt (all stored in torch.int8), for deriving expected scales.
QMAX = {fmt: _STR_TO_QMAX[fmt] for fmt in sorted(_INT8_FORMATS)}


def _fp8_kw(scale_dtype=None):
    return dict(fmt="fp8_e4m3", scale_dtype=scale_dtype)


def _int_kw(fmt="int8"):
    return dict(fmt=fmt)


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
    assert xq.shape == (8, 256) and xq.dtype == E4M3
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


# --- fake-quant: reconstruction, dtype, zero-safety ---


def test_fake_quant_reconstructs_within_fp8_error():
    torch.manual_seed(0)
    x = torch.randn(16, 128)
    fq = fake_quantize_operand(x, -1, "rowwise", 128, **_fp8_kw())
    assert fq.dtype == x.dtype
    assert (fq - x).norm() / x.norm() < 0.1


def test_tensorwise_fp8_roundtrips_within_error():
    torch.manual_seed(0)
    x = torch.randn(64, 64) * 10.0
    xq, _ = quantize_operand(x, -1, "tensorwise", 0, **_fp8_kw())
    assert xq.dtype == E4M3
    recon = fake_quantize_operand(x, -1, "tensorwise", 0, **_fp8_kw())
    assert (recon - x).norm() / x.norm() < 0.1  # e4m3 per-tensor dynamic range


@pytest.mark.parametrize("gran,bs", [("tensorwise", 0), ("blockwise", 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fake_quant_preserves_dtype_and_shape(gran, bs, dtype):
    x = torch.randn(8, 64, dtype=dtype)
    out = fake_quantize_operand(x, -1, gran, bs, **_fp8_kw())
    assert out.dtype == dtype and out.shape == x.shape


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
def test_zero_input_safe(gran, bs):
    x = torch.zeros(4, 64)
    fq = fake_quantize_operand(x, -1, gran, bs, **_fp8_kw())
    assert torch.isfinite(fq).all()


# --- int operand quantize ---


def test_int_quant_rounds_and_clamps():
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 10
    xq, s = quantize_operand(x, -1, "rowwise", 64, **_int_kw("int4"))
    assert xq.dtype == torch.int8
    assert int(xq.min()) >= -7 and int(xq.max()) <= 7


@pytest.mark.parametrize("fmt", list(QMAX))
def test_int_tensorwise_symmetric_range_and_scale(fmt):
    torch.manual_seed(0)
    qmax = QMAX[fmt]
    x = torch.randn(64, 64) * 10.0
    q, s = quantize_operand(x, -1, "tensorwise", 0, **_int_kw(fmt))
    assert q.dtype == torch.int8  # all widths stored in int8
    assert int(q.min()) >= -qmax and int(q.max()) <= qmax
    # tensorwise scale broadcasts a single value = amax / qmax over the rows
    assert torch.allclose(s, x.abs().max().float() / qmax)


@pytest.mark.parametrize("fmt", list(QMAX))
def test_int_tensorwise_recon_within_half_scale(fmt):
    torch.manual_seed(0)
    qmax = QMAX[fmt]
    x = torch.randn(128, 128)
    recon = fake_quantize_operand(x, -1, "tensorwise", 0, **_int_kw(fmt))
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
    assert xq.dtype == E4M3 and s.shape == (4, 2)
    log2s = torch.log2(s[s > 0])
    assert torch.allclose(log2s, log2s.round())


def test_pow2_fake_quant_reasonable_error():
    torch.manual_seed(0)
    x = torch.randn(8, 128)
    fq = fake_quantize_operand(
        x, -1, "blockwise", 32, **_fp8_kw(scale_dtype="fp8_e8m0")
    )
    assert (fq - x).norm() / x.norm() < 0.15


# --- partial-block bs recovery regression ---


def _manual_dequant(x, contract_dim, bs, *, fmt, scale_dtype=None):
    """Reference dequant using the *true* block size `bs` (never inferred from
    `nkb`/`K`), to regression-test that fake_quantize_operand's own bs recovery
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


def test_fake_quant_partial_block_matches_manual_dequant_operand_a():
    # Regression test for a bug where fake_quantize_operand reconstructed the
    # block size as ceil(K/nkb) instead of the true block_size, which diverges
    # whenever the last block is partial (K=100, bs=32 -> nkb=4, but
    # ceil(100/4)=25 != 32). Would FAIL under that buggy reconstruction.
    torch.manual_seed(0)
    x = torch.randn(4, 100)
    fq = fake_quantize_operand(x, -1, "blockwise", 32, **_fp8_kw())
    manual = _manual_dequant(x, -1, 32, **_fp8_kw()).to(x.dtype)
    assert torch.allclose(fq, manual, atol=1e-6, rtol=1e-5)


def test_fake_quant_partial_block_matches_manual_dequant_operand_b():
    # Same regression, operand B (contract_dim=0), K=100 along dim 0.
    torch.manual_seed(0)
    x = torch.randn(100, 6)
    fq = fake_quantize_operand(x, 0, "blockwise", 32, **_fp8_kw())
    manual = _manual_dequant(x, 0, 32, **_fp8_kw()).to(x.dtype)
    assert torch.allclose(fq, manual, atol=1e-6, rtol=1e-5)
