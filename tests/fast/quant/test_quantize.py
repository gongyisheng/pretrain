import pytest
import torch

from src.quant.constants import _INT8_FORMATS
from src.quant.quantize import (
    quantize_operand,
    dequantize_operand,
    _scale_block_map,
)
from src.quant.utils import str_to_dtype, str_to_qmax

_E4M3 = "fp8_e4m3"


def _scale(gran, bs=0, scale_dtype=torch.float32):
    return {
        "granularity": gran,
        "block_shape": (1, bs) if bs else (0, 0),
        "scale_dtype": scale_dtype,
    }


def _roundtrip(x, contract_dim, fmt, scale_cfg):
    """dequant(quant(x)) — the reconstruction the quantized GEMMs actually see."""
    xq, scale = quantize_operand(x, contract_dim, fmt, scale_cfg)
    return dequantize_operand(xq, scale, contract_dim, scale_cfg)


def test_quantize_requires_scale_dtype():
    with pytest.raises(KeyError, match="scale_dtype"):
        quantize_operand(
            torch.ones(2, 2),
            -1,
            _E4M3,
            {"granularity": "rowwise", "block_shape": (0, 0)},
        )


@pytest.mark.parametrize("fmt", ["int4", "fp8_e4m3", "fp8_e5m2"])
def test_stochastic_rounding_is_unbiased_between_adjacent_codes(fmt):
    # The leading 1 sets the row scale; .3 is strictly between two codes for every
    # format here, so its expected reconstruction is exactly .3.
    x = torch.cat([torch.ones(1), torch.full((9999,), 0.3)]).reshape(1, -1)
    codes, scale = quantize_operand(
        x, -1, fmt, _scale("rowwise"), stochastic_rounding=True
    )
    reconstructed = codes.float() * scale
    rounded = codes.float()[0, 1:]

    assert rounded.unique().numel() == 2
    assert reconstructed[0, 1:].mean().item() == pytest.approx(0.3, abs=0.002)


@pytest.mark.parametrize("fmt", ["int4", "fp8_e4m3", "fp8_e5m2"])
def test_stochastic_rounding_preserves_exact_endpoints(fmt):
    x = torch.tensor([[1.0, 0.0, -0.0, -1.0]])
    codes, _ = quantize_operand(x, -1, fmt, _scale("rowwise"), stochastic_rounding=True)

    qmax = str_to_qmax(fmt)
    assert torch.equal(codes.float(), torch.tensor([[qmax, 0.0, -0.0, -qmax]]))
    if fmt.startswith("fp8"):
        assert torch.signbit(codes.float()[0, 2])


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_stochastic_rounding_never_leaves_the_adjacent_codes(fmt):
    """The gate on the per-element ULP: a wrong grid lands on a non-adjacent code.

    Probes every code, every midpoint between codes, and the subnormal run below the
    smallest normal, which shares one spacing instead of halving per binade.
    """
    dtype = str_to_dtype(fmt)
    codes = torch.arange(256, dtype=torch.uint8).view(dtype).float()
    codes = codes[codes.isfinite()].unique()
    x = torch.cat([codes, (codes[:-1] + codes[1:]) / 2]).reshape(1, -1)

    # scale 1: the row's amax is qmax, so the codes are the values themselves
    rounded, scale = quantize_operand(
        x, -1, fmt, _scale("rowwise"), stochastic_rounding=True
    )
    assert scale.item() == 1.0
    below = torch.searchsorted(codes, x.flatten().contiguous())
    lower = codes[(below - 1).clamp(min=0)]
    upper = codes[below.clamp(max=codes.numel() - 1)]
    got = rounded.float().flatten()
    assert bool(((got == lower) | (got == upper)).all())


def test_deterministic_rounding_does_not_consume_rng():
    x = torch.tensor([[1.0, 0.3]])
    torch.manual_seed(0)
    before = torch.random.get_rng_state()
    quantize_operand(x, -1, _E4M3, _scale("rowwise"))
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("bs", [16, 32, 128])
@pytest.mark.parametrize("contract_dim", [-1, -2])
def test_block_shape_1d_matches_legacy_block_size(bs, contract_dim):
    """(1, B) is exactly what block_size=B produced. The gate on the rename."""
    x = torch.randn(64, 128)
    codes, scale = quantize_operand(
        x,
        contract_dim,
        _E4M3,
        {
            "granularity": "blockwise",
            "block_shape": (1, bs),
            "scale_dtype": torch.float32,
        },
    )
    expected_blocks = (x.shape[contract_dim] + bs - 1) // bs
    assert scale.shape[contract_dim] == expected_blocks
    outer_dim = -2 if contract_dim == -1 else -1
    assert scale.shape[outer_dim] == x.shape[outer_dim]


# --- ragged scale blocks ---


def _offs(counts):
    return torch.tensor(counts).cumsum(0).to(torch.int32)


def test_scale_block_map_one_block_per_group():
    # block_size 0 — what row/tensorwise normalize to — is one block per group
    row_blocks, n_blocks = _scale_block_map(_offs([3, 4, 3]), 10, 0)
    assert row_blocks.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    assert n_blocks == 3


def test_scale_block_map_blockwise_retiles_inside_each_group():
    # bs=4 with group sizes 3,4,3: every group starts a fresh block, so no block
    # spans a boundary even though 3 and 4 are not multiples of each other.
    row_blocks, _ = _scale_block_map(_offs([3, 4, 3]), 10, 4)
    assert row_blocks.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]


def test_scale_block_map_blockwise_multiple_blocks_and_empty_group():
    # counts 5,0,4 with bs=2 -> group 0 gets 3 blocks, group 1 none, group 2 two.
    row_blocks, n_blocks = _scale_block_map(_offs([5, 0, 4]), 9, 2)
    assert row_blocks.tolist() == [0, 0, 1, 1, 2, 3, 3, 4, 4]
    assert n_blocks == 9 // 2 + 3


def test_scale_block_map_never_crosses_a_group_boundary():
    counts = [5, 0, 130, 41, 1]
    offs = _offs(counts)
    n_rows = sum(counts)
    row_blocks, _ = _scale_block_map(offs, n_rows, 32)
    group = torch.searchsorted(
        offs, torch.arange(n_rows, dtype=torch.int32), right=True
    )
    # a block id must map to exactly one group
    for block_id in row_blocks.unique().tolist():
        assert group[row_blocks == block_id].unique().numel() == 1


@pytest.mark.parametrize(
    "counts,bs",
    [([5, 0, 130, 41, 1], 32), ([300, 5, 120, 0, 44, 210], 128), ([1, 1, 1], 16)],
)
def test_scale_block_map_n_blocks_is_a_valid_upper_bound(counts, bs):
    row_blocks, n_blocks = _scale_block_map(_offs(counts), sum(counts), bs)
    assert int(row_blocks.max()) < n_blocks


# --- operand quantize: canonical scale shapes ---


@pytest.mark.parametrize(
    "gran,bs,nkb", [("tensorwise", 0, 1), ("rowwise", 0, 1), ("blockwise", 64, 4)]
)
def test_operand_a_scale_shape(gran, bs, nkb):
    x = torch.randn(8, 256)
    xq, s = quantize_operand(x, -1, _E4M3, _scale(gran, bs))
    assert xq.shape == (8, 256) and xq.dtype == torch.float8_e4m3fn
    assert s.shape == (8, nkb) and s.dtype == torch.float32 and s.is_contiguous()


@pytest.mark.parametrize(
    "gran,bs,nkb", [("tensorwise", 0, 1), ("rowwise", 0, 1), ("blockwise", 64, 4)]
)
def test_operand_b_scale_shape(gran, bs, nkb):
    x = torch.randn(256, 12)
    xq, s = quantize_operand(x, -2, _E4M3, _scale(gran, bs))
    assert xq.shape == (256, 12)
    assert s.shape == (nkb, 12) and s.is_contiguous()


def test_blockwise_partial_last_block():
    # K=100 not a multiple of 32 -> nkb=4, last block covers [96,100)
    x = torch.randn(4, 100)
    xq, s = quantize_operand(x, -1, _E4M3, _scale("blockwise", 32))
    assert xq.shape == (4, 100) and s.shape == (4, 4)


# --- roundtrip error, input dtypes, zero-safety ---


def test_rowwise_fp8_roundtrips_within_error():
    torch.manual_seed(0)
    x = torch.randn(16, 128)
    fq = _roundtrip(x, -1, _E4M3, _scale("rowwise"))
    assert (fq - x).norm() / x.norm() < 0.1


def test_tensorwise_fp8_roundtrips_within_error():
    torch.manual_seed(0)
    x = torch.randn(64, 64) * 10.0
    xq, _ = quantize_operand(x, -1, _E4M3, _scale("tensorwise"))
    assert xq.dtype == torch.float8_e4m3fn
    recon = _roundtrip(x, -1, _E4M3, _scale("tensorwise"))
    assert (recon - x).norm() / x.norm() < 0.1  # e4m3 per-tensor dynamic range


@pytest.mark.parametrize("gran,bs", [("tensorwise", 0), ("blockwise", 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_quantize_operand_accepts_all_compute_dtypes(gran, bs, dtype):
    x = torch.randn(8, 64, dtype=dtype)
    xq, s = quantize_operand(x, -1, _E4M3, _scale(gran, bs))
    assert xq.shape == x.shape and xq.dtype == torch.float8_e4m3fn
    assert s.dtype == torch.float32  # scale is fp32 regardless of input dtype


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
def test_zero_input_safe(gran, bs):
    x = torch.zeros(4, 64)
    fq = _roundtrip(x, -1, _E4M3, _scale(gran, bs))
    assert torch.isfinite(fq).all()


# --- int operand quantize ---


def test_int_quant_rounds_and_clamps():
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 10
    xq, s = quantize_operand(x, -1, "int4", _scale("rowwise"))
    assert xq.dtype == torch.int8
    assert int(xq.min()) >= -7 and int(xq.max()) <= 7


@pytest.mark.parametrize("fmt", sorted(_INT8_FORMATS))
def test_int_tensorwise_symmetric_range_and_scale(fmt):
    torch.manual_seed(0)
    qmax = str_to_qmax(fmt)
    x = torch.randn(64, 64) * 10.0
    q, s = quantize_operand(x, -1, fmt, _scale("tensorwise"))
    assert q.dtype == torch.int8  # all widths stored in int8
    assert int(q.min()) >= -qmax and int(q.max()) <= qmax
    # tensorwise scale broadcasts a single value = amax / qmax over the rows
    assert torch.allclose(s, x.abs().max().float() / qmax)


@pytest.mark.parametrize("fmt", sorted(_INT8_FORMATS))
def test_int_tensorwise_recon_within_half_scale(fmt):
    torch.manual_seed(0)
    qmax = str_to_qmax(fmt)
    x = torch.randn(128, 128)
    recon = _roundtrip(x, -1, fmt, _scale("tensorwise"))
    scale = x.abs().max().float() / qmax
    assert (recon - x).abs().max() <= scale.item() / 2 + 1e-6


# --- power-of-two (E8M0 / MX-style) scale ---

_MXFP8 = _scale("blockwise", 32, torch.float8_e8m0fnu)


def test_power_of_two_scale_is_exact_pow2():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    _, s = quantize_operand(x, -1, _E4M3, _MXFP8)
    log2s = torch.log2(s[s > 0])
    assert torch.allclose(log2s, log2s.round())


def test_pow2_scale_block32_shape_and_exactness():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    xq, s = quantize_operand(x, -1, _E4M3, _MXFP8)
    assert xq.dtype == torch.float8_e4m3fn and s.shape == (4, 2)
    log2s = torch.log2(s[s > 0])
    assert torch.allclose(log2s, log2s.round())


def test_pow2_scale_roundtrips_within_error():
    torch.manual_seed(0)
    x = torch.randn(8, 128)
    fq = _roundtrip(x, -1, _E4M3, _MXFP8)
    assert (fq - x).norm() / x.norm() < 0.15


@pytest.mark.parametrize(
    ("fmt", "value", "code"),
    [("fp8_e4m3", 2**-136, 2**-9), ("fp8_e5m2", 2**-143, 2**-16)],
)
def test_e8m0_tiny_nonzero_block_uses_minimum_scale(fmt, value, code):
    x = torch.zeros(1, 32)
    x[0, 0] = value
    xq, scale = quantize_operand(x, -1, fmt, _MXFP8)

    assert torch.equal(scale, torch.full_like(scale, 2**-127))
    assert xq.float()[0, 0] == code
    assert torch.count_nonzero(xq) == 1
    assert torch.equal(dequantize_operand(xq, scale, -1, _MXFP8), x)


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_e8m0_zero_block_uses_minimum_scale(fmt):
    x = torch.zeros(1, 32)
    xq, scale = quantize_operand(x, -1, fmt, _MXFP8)

    assert torch.equal(scale, torch.full_like(scale, 2**-127))
    assert torch.count_nonzero(xq) == 0
    assert torch.count_nonzero(dequantize_operand(xq, scale, -1, _MXFP8)) == 0


@pytest.mark.parametrize("granularity", ["tensorwise", "rowwise"])
def test_fp32_scale_never_clips(granularity):
    # scale = amax/qmax lands the extremal element exactly on qmax, so the clamp in
    # quantize_operand is a no-op and the largest magnitude round-trips intact.
    torch.manual_seed(0)
    x = torch.randn(32, 64) * 10.0
    fq = _roundtrip(x, -1, _E4M3, _scale(granularity))
    assert fq.abs().amax().item() == pytest.approx(x.abs().amax().item(), rel=1e-6)


def test_e8m0_scale_does_not_saturate_the_block_maximum():
    # The exponent is picked against qmax, not emax, so amax/scale <= qmax always:
    # 1.9 -> ceil(log2(1.9/448)) = -7 -> 1.9 / 2**-7 = 243.2, well inside e4m3.
    x = torch.full((1, 32), 1.9)
    xq, s = quantize_operand(x, -1, _E4M3, _MXFP8)
    assert xq.float().abs().amax().item() < str_to_qmax(_E4M3)
    # 243.2 rounds to the nearest e4m3 code, 240, so 1.9 comes back as 240 * 2**-7.
    fq = dequantize_operand(xq, s, -1, _MXFP8)
    assert torch.allclose(fq, torch.full_like(fq, 1.875))


def test_e8m0_block_maximum_roundtrips_within_half_an_ulp():
    # With no clipping the peak of each block loses only e4m3 round-to-nearest:
    # 3 mantissa bits, so at most half an ulp = 2**-4 relative.
    torch.manual_seed(0)
    x = (torch.randn(64, 256) * 3.0).reshape(64, 8, 32)
    fq = _roundtrip(x.reshape(64, 256), -1, _E4M3, _MXFP8).reshape(64, 8, 32)
    idx = x.abs().argmax(-1, keepdim=True)
    peak, got = x.gather(-1, idx).abs(), fq.gather(-1, idx).abs()
    assert ((peak - got) / peak).max().item() <= 2**-4


# --- scale values: what each scale group actually reduces ---


def _block_amax(x, contract_dim, bs):
    """Per-group amax along `contract_dim`, tiled with plain tensor ops.

    Deliberately not built from `_tile`/`_scale_block_map`: this is the independent
    side of the comparison, so it must not share the helpers under test. amax is an
    exact reduction, so equality against the implementation is bitwise.
    """
    v = x if contract_dim == -1 else x.mT
    K = v.shape[-1]
    width = bs or K
    n_blocks = (K + width - 1) // width
    pad = n_blocks * width - K
    if pad:
        v = torch.nn.functional.pad(v, (0, pad))
    amax = v.abs().reshape(*v.shape[:-1], n_blocks, width).amax(-1)
    return amax if contract_dim == -1 else amax.mT


@pytest.mark.parametrize("contract_dim", [-1, -2])
@pytest.mark.parametrize("bs", [0, 32, 40])  # 0 is rowwise; 40 leaves a short block
def test_scale_reduces_exactly_its_own_block(contract_dim, bs):
    torch.manual_seed(0)
    x = torch.randn(64, 96) if contract_dim == -1 else torch.randn(96, 64)
    gran = "rowwise" if bs == 0 else "blockwise"
    _, s = quantize_operand(x, contract_dim, _E4M3, _scale(gran, bs))
    assert torch.equal(s, _block_amax(x, contract_dim, bs) / str_to_qmax(_E4M3))


@pytest.mark.parametrize("contract_dim", [-1, -2])
def test_e8m0_scale_lands_block_amax_in_the_top_binade(contract_dim):
    # 2**ceil(log2(amax/qmax)) puts amax/scale in (qmax/2, qmax]: never above, so no
    # code clips, and never a full binade below, so no range is wasted.
    torch.manual_seed(0)
    x = (torch.randn(64, 96) if contract_dim == -1 else torch.randn(96, 64)) * 7.0
    _, s = quantize_operand(x, contract_dim, _E4M3, _MXFP8)
    qmax = str_to_qmax(_E4M3)
    ratio = _block_amax(x, contract_dim, 32) / s
    assert (ratio <= qmax).all() and (ratio > qmax / 2).all()


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
def test_expert_stack_scale_matches_per_expert_quantize(gran, bs):
    # One rank-agnostic call on an (E,K,N) slab must agree with quantizing expert by
    # expert — the scale layout for a 3D operand is only exercised here.
    torch.manual_seed(0)
    b = torch.randn(3, 96, 32)
    codes, scale = quantize_operand(b, -2, _E4M3, _scale(gran, bs))
    per = [quantize_operand(b[g], -2, _E4M3, _scale(gran, bs)) for g in range(3)]
    assert torch.equal(
        codes.view(torch.uint8), torch.stack([q for q, _ in per]).view(torch.uint8)
    )
    assert torch.equal(scale, torch.stack([s for _, s in per]))


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 4)]
)
def test_ragged_scale_blocks_match_independent_per_group_quantize(gran, bs):
    # Codes are pinned by test_ragged_quantize_matches_independent_per_group_quantize;
    # this pins the scales they were divided by, which is where cross-group pooling
    # would actually show up.
    torch.manual_seed(0)
    counts = [6, 0, 5]
    x = torch.randn(sum(counts), 8)
    x[:6] *= 50.0
    offs, scale_cfg = _offs(counts), _scale(gran, bs)
    _, scale = quantize_operand(x, -2, _E4M3, scale_cfg, offs=offs, ragged_dim=-2)
    row_blocks, _ = _scale_block_map(offs, sum(counts), bs)
    lo = 0
    for hi in offs.tolist():
        if hi > lo:
            _, expected = quantize_operand(x[lo:hi], -2, _E4M3, scale_cfg)
            assert torch.equal(scale[row_blocks[lo:hi].unique()], expected)
        lo = hi


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 4)]
)
def test_ragged_last_dim_is_the_transpose_of_ragged_first(gran, bs):
    """Quantizing the transposed view == transposing the quantization.

    This is what would let a caller whose operand already arrives transposed quantize
    it where it stands instead of flipping it in and back out again. `moe.py`'s wgrad
    is that caller and deliberately does not: reducing along the strided axis of the
    view costs more than the transposed kernel read it would save (measured 1.2x-2.8x
    on the quantize path). The equivalence is pinned here regardless -- it is the
    property `ragged_dim` has to have.
    """
    torch.manual_seed(0)
    x = torch.randn(24, 16)
    offs, scale_cfg = _offs([8, 0, 10, 6]), _scale(gran, bs)
    codes, scale = quantize_operand(x, -2, _E4M3, scale_cfg, offs=offs, ragged_dim=-2)
    codes_t, scale_t = quantize_operand(
        x.mT, -1, _E4M3, scale_cfg, offs=offs, ragged_dim=-1
    )
    assert torch.equal(
        codes_t.view(torch.uint8), codes.mT.contiguous().view(torch.uint8)
    )
    assert torch.equal(scale_t, scale.mT.contiguous())


# --- dequantize_operand ---


def _manual_dequant(x, contract_dim, bs, fmt, scale_dtype=torch.float32):
    """Reference dequant using the *true* block size `bs` (never inferred from
    `nkb`/`K`), to regression-test that dequantize_operand's own bs recovery
    matches it even when the last block is partial (K not a multiple of bs)."""
    xq, s = quantize_operand(x, contract_dim, fmt, _scale("blockwise", bs, scale_dtype))
    qf = xq.movedim(contract_dim, -1).float()
    K = qf.shape[-1]
    sf = s.movedim(contract_dim, -1).float()
    nkb = sf.shape[-1]
    pad = nkb * bs - K
    qp = torch.nn.functional.pad(qf, (0, pad)) if pad else qf
    deq = (qp.unflatten(-1, (nkb, bs)) * sf.unsqueeze(-1)).flatten(-2)[..., :K]
    return deq.movedim(-1, contract_dim)


@pytest.mark.parametrize("contract_dim", [-1, -2])
@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
def test_dequantize_operand_preserves_shape_and_layout(gran, bs, contract_dim):
    torch.manual_seed(0)
    x = torch.randn(64, 96)
    xq, s = quantize_operand(x, contract_dim, _E4M3, _scale(gran, bs))
    deq = dequantize_operand(xq, s, contract_dim, _scale(gran, bs))
    assert deq.shape == x.shape and deq.is_contiguous()


def test_dequantize_operand_returns_float32_from_bf16():
    # metrics compare in fp32 regardless of the operand's compute dtype
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    xq, s = quantize_operand(x, -1, _E4M3, _scale("rowwise"))
    assert dequantize_operand(xq, s, -1, _scale("rowwise")).dtype == torch.float32


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
)
def test_dequantize_operand_stacked_experts_matches_per_expert(gran, bs):
    # The MoE path quantizes an (E,K,N) slab along dim -2 and inverts it in one call.
    # Assert that agrees with dequantizing slab by slab — the two paths are written
    # differently, and the per-expert list is also what the deleted loop produced.
    torch.manual_seed(0)
    experts = torch.randn(4, 96, 32)
    per_expert = [
        quantize_operand(experts[g], -2, _E4M3, _scale(gran, bs))
        for g in range(experts.shape[0])
    ]
    deq = dequantize_operand(
        torch.stack([q for q, _ in per_expert]),
        torch.stack([s for _, s in per_expert]),
        -2,
        _scale(gran, bs),
    )
    reference = torch.stack(
        [
            _roundtrip(experts[g], -2, _E4M3, _scale(gran, bs)).float()
            for g in range(experts.shape[0])
        ]
    )
    assert torch.equal(deq, reference)


def test_dequantize_operand_partial_block_uses_given_block_size():
    # K=100, bs=32 -> nkb=4; an inferred ceil(100/4)=25 would diverge.
    torch.manual_seed(0)
    x = torch.randn(4, 100)
    scale_cfg = _scale("blockwise", 32)
    xq, s = quantize_operand(x, -1, _E4M3, scale_cfg)
    deq = dequantize_operand(xq, s, -1, scale_cfg)
    assert torch.allclose(deq, _manual_dequant(x, -1, 32, _E4M3), atol=1e-6)


def test_dequantize_operand_rejects_a_contraction_outside_the_last_two_dims():
    # contract_dim names one of the last two dims, so 0 is outside what the tiling
    # helpers split. It must raise, not tile the wrong axis at a plausible shape.
    xq = torch.ones(3, 3, 4, dtype=torch.int8)
    scale = torch.arange(1.0, 13.0).reshape(1, 3, 4)
    with pytest.raises(ValueError, match="dim must be -2 or -1"):
        dequantize_operand(xq, scale, 0, _scale("tensorwise"))


def test_quantize_operand_rejects_a_contraction_outside_the_last_two_dims():
    with pytest.raises(ValueError, match="contract_dim must be -2 or -1"):
        quantize_operand(torch.randn(4, 8), 0, _E4M3, _scale("tensorwise"))


def test_quantize_operand_rejects_offs_without_ragged_dim():
    with pytest.raises(ValueError, match="together"):
        quantize_operand(
            torch.randn(4, 8), -1, _E4M3, _scale("tensorwise"), offs=_offs([2, 2])
        )


def test_quantize_operand_rejects_a_ragged_axis_on_a_stacked_operand():
    with pytest.raises(ValueError, match="2D operand"):
        quantize_operand(
            torch.randn(2, 4, 8),
            -2,
            _E4M3,
            _scale("tensorwise"),
            offs=_offs([2, 2]),
            ragged_dim=-2,
        )


# --- ragged contraction axis (wgrad): per-expert scales ---


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 4)]
)
def test_ragged_quantize_scale_shape_and_bound(gran, bs):
    counts = [6, 0, 5]
    offs = _offs(counts)
    _, n_blocks = _scale_block_map(offs, sum(counts), bs)
    x = torch.randn(sum(counts), 8)
    xq, scale = quantize_operand(
        x, -2, _E4M3, _scale(gran, bs), offs=offs, ragged_dim=-2
    )
    assert xq.shape == x.shape
    assert scale.shape == (n_blocks, 8)
    assert scale.dtype == torch.float32


def test_ragged_quantize_cold_group_scale_ignores_hot_group():
    # The bug this whole change exists to fix: group 0 is 100x hotter than group 1.
    # A shared amax would scale both by group 0's, wiping out group 1's codes.
    x = torch.cat([torch.full((4, 3), 100.0), torch.full((4, 3), 1.0)])
    _, scale = quantize_operand(
        x, -2, _E4M3, _scale("rowwise"), offs=_offs([4, 4]), ragged_dim=-2
    )
    assert torch.allclose(scale[0] / scale[1], torch.full((3,), 100.0), rtol=1e-5)


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 4)]
)
def test_ragged_quantize_matches_independent_per_group_quantize(gran, bs):
    # Each group quantized on its own must give byte-identical codes to the
    # fused ragged call — that IS the definition of expert independence.
    torch.manual_seed(0)
    counts = [6, 0, 5]
    x = torch.randn(sum(counts), 8)
    x[:6] *= 50.0  # make the groups differ enough that a shared amax would show
    offs = _offs(counts)
    xq, _ = quantize_operand(x, -2, _E4M3, _scale(gran, bs), offs=offs, ragged_dim=-2)
    lo = 0
    for hi in offs.tolist():
        if hi > lo:
            # a single group is a non-ragged tensor: the existing code path
            expected, _ = quantize_operand(x[lo:hi], -2, _E4M3, _scale(gran, bs))
            assert torch.equal(xq[lo:hi], expected)
        lo = hi


def test_ragged_quantize_empty_group_scale_is_finite():
    _, scale = quantize_operand(
        torch.randn(4, 3),
        -2,
        _E4M3,
        _scale("rowwise"),
        offs=_offs([4, 0]),
        ragged_dim=-2,
    )
    assert torch.isfinite(scale).all()
    assert (scale > 0).all()


@pytest.mark.parametrize(
    "contract_dim,gran,bs",
    [
        (-1, "rowwise", 0),
        (-1, "blockwise", 32),
        (-2, "rowwise", 0),
        (-2, "blockwise", 32),
    ],
)
def test_ragged_none_is_unchanged(contract_dim, gran, bs):
    # QuantizedLinear passes no `offs`; that path must stay bit-identical.
    torch.manual_seed(0)
    x = torch.randn(64, 96)
    xq, scale = quantize_operand(x, contract_dim, _E4M3, _scale(gran, bs))
    assert xq.shape == x.shape
    expected_nkb = 1 if bs == 0 else (96 if contract_dim == -1 else 64) // bs
    assert scale.shape[contract_dim] == expected_nkb


@pytest.mark.parametrize(
    "gran,bs", [("tensorwise", 0), ("rowwise", 0), ("blockwise", 4)]
)
def test_ragged_dequantize_inverts_ragged_quantize(gran, bs):
    torch.manual_seed(0)
    counts = [6, 0, 5]
    x = torch.randn(sum(counts), 8)
    x[:6] *= 50.0
    offs = _offs(counts)
    scale_cfg = _scale(gran, bs)
    xq, scale = quantize_operand(x, -2, _E4M3, scale_cfg, offs=offs, ragged_dim=-2)
    deq = dequantize_operand(xq, scale, -2, scale_cfg, offs=offs, ragged_dim=-2)
    assert deq.shape == x.shape
    assert deq.dtype == torch.float32
    # per group, because a shared reduction would hide a mis-scaled cold group
    lo = 0
    for hi in offs.tolist():
        if hi > lo:
            rel = (deq[lo:hi] - x[lo:hi]).norm() / x[lo:hi].norm()
            assert rel < 0.1, (lo, hi, rel)
        lo = hi


# --- ragged row axis (fwd/dgrad): only tensorwise pooled across groups ---


def test_ragged_tensorwise_row_scale_is_per_group():
    x = torch.cat([torch.full((4, 3), 100.0), torch.full((4, 3), 1.0)])
    xq, scale = quantize_operand(
        x, -1, _E4M3, _scale("tensorwise"), offs=_offs([4, 4]), ragged_dim=-2
    )
    assert scale.shape == (8, 1)  # canonical A layout, unchanged
    assert torch.allclose(scale[:4], scale[0].expand(4, 1))
    assert torch.allclose(scale[0] / scale[4], torch.tensor([100.0]), rtol=1e-5)
    # the cold group keeps its full code range instead of collapsing to ~1/100
    assert xq[4:].float().abs().max() > 0.5 * xq[:4].float().abs().max()


@pytest.mark.parametrize("gran,bs", [("rowwise", 0), ("blockwise", 4)])
def test_ragged_ignored_for_already_per_row_granularities(gran, bs):
    torch.manual_seed(0)
    x = torch.randn(8, 8)
    with_ragged = quantize_operand(
        x, -1, _E4M3, _scale(gran, bs), offs=_offs([4, 4]), ragged_dim=-2
    )
    without = quantize_operand(x, -1, _E4M3, _scale(gran, bs))
    assert torch.equal(with_ragged[0], without[0])
    assert torch.equal(with_ragged[1], without[1])


@pytest.mark.parametrize("shape", [(64, 128), (70, 130), (4, 64, 128)])
@pytest.mark.parametrize("tile", [16, 32])
def test_tile2d_roundtrip(shape, tile):
    from src.quant.quantize import _tile2d, _untile2d

    x = torch.randn(*shape)
    rows, cols = shape[-2:]
    v = _tile2d(x, tile)
    assert v.shape[-4:] == (
        (rows + tile - 1) // tile,
        tile,
        (cols + tile - 1) // tile,
        tile,
    )
    assert torch.equal(_untile2d(v, rows, cols), x)


@pytest.mark.parametrize("tile", [16, 32])
def test_tile2d_amax_reduces_to_block_counts(tile):
    from src.quant.quantize import _tile2d

    x = torch.randn(64, 128)
    amax = _tile2d(x, tile).abs().amax((-3, -1))
    assert amax.shape == (64 // tile, 128 // tile)
    # a tile's amax is the max over exactly that square
    assert torch.equal(amax[1, 2], x[tile : 2 * tile, 2 * tile : 3 * tile].abs().amax())


# --- 2D (square-tile) blockwise ---


def _scale2d(tile, scale_dtype=torch.float32):
    return {
        "granularity": "blockwise",
        "block_shape": (tile, tile),
        "scale_dtype": scale_dtype,
    }


@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("shape", [(64, 128), (70, 130)])
def test_blockwise_2d_is_transpose_invariant(tile, shape):
    """The property the whole feature exists for: a square tile quantizes W and W.T
    to the same codes, so the forward and dgrad see the same weight."""
    w = torch.randn(*shape)
    codes_a, _ = quantize_operand(w, -1, _E4M3, _scale2d(tile))
    codes_b, _ = quantize_operand(w.t().contiguous(), -2, _E4M3, _scale2d(tile))
    assert torch.equal(codes_a.t().float(), codes_b.float())


@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("contract_dim", [-1, -2])
def test_blockwise_2d_scale_uses_the_1d_layout(tile, contract_dim):
    """No kernel may need to know the tile was square: the scale leaves in the same
    shape a 1D blockwise scale of the same contract extent would."""
    x = torch.randn(64, 128)
    _, scale_2d = quantize_operand(x, contract_dim, _E4M3, _scale2d(tile))
    _, scale_1d = quantize_operand(
        x, contract_dim, _E4M3, _scale("blockwise", tile, torch.float32)
    )
    assert scale_2d.shape == scale_1d.shape


@pytest.mark.parametrize("tile", [16, 32])
def test_blockwise_2d_scale_is_constant_within_a_tile(tile):
    """Guards the expansion direction -- the one orientation-dependent step."""
    x = torch.randn(64, 128)
    _, scale = quantize_operand(x, -1, _E4M3, _scale2d(tile))  # (64, 128/tile)
    for i in range(64 // tile):
        block = scale[i * tile : (i + 1) * tile, :]
        assert torch.equal(block, block[:1].expand_as(block))


@pytest.mark.parametrize("shape", [(64, 128), (70, 130), (4, 64, 128)])
def test_blockwise_2d_roundtrip(shape):
    x = torch.randn(*shape)
    scale_cfg = _scale2d(32)
    deq = _roundtrip(x, -1, _E4M3, scale_cfg)
    assert deq.shape == x.shape
    assert (deq - x).abs().max() < 0.1 * x.abs().max()


def test_blockwise_2d_rejects_a_ragged_axis():
    x = torch.randn(64, 128)
    offs = _offs([32, 64])
    with pytest.raises(NotImplementedError, match="ragged"):
        quantize_operand(x, -2, _E4M3, _scale2d(16), offs=offs, ragged_dim=-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="2D quantizer compilation requires CUDA"
)
@pytest.mark.parametrize("stochastic_rounding", [False, True])
def test_quantize_operand_2d_compiles_fullgraph(stochastic_rounding):
    scale_cfg = {
        "granularity": "blockwise",
        "block_shape": (32, 32),
        "scale_dtype": torch.float32,
    }
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    fn = torch.compile(
        lambda: quantize_operand(
            x, -1, _E4M3, scale_cfg, stochastic_rounding=stochastic_rounding
        ),
        fullgraph=True,
    )
    _, scale = fn()
    assert scale.shape == (256, 16)
