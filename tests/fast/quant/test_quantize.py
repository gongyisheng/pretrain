import pytest
import torch

from src.quant.constants import _INT8_FORMATS
from src.quant.quantize import (
    _scale_block_map,
    _tile2d,
    _untile2d,
    dequantize_operand,
    quantize_operand,
)
from src.quant.utils import str_to_dtype, str_to_qmax

E4M3 = "fp8_e4m3"
FP8_FORMATS = [E4M3, "fp8_e5m2"]
INT_FORMATS = sorted(_INT8_FORMATS)
# int4 and both fp8 grids are the three distinct rounding lattices
ROUNDING_FORMATS = ["int4", E4M3, "fp8_e5m2"]

CONTRACT_DIMS = [-1, -2]
TILES = [16, 32]


def _scale(granularity, block=0, scale_dtype=torch.float32):
    return {
        "granularity": granularity,
        "block_shape": (1, block) if block else (0, 0),
        "scale_dtype": scale_dtype,
    }


def _scale2d(tile, scale_dtype=torch.float32):
    return {
        "granularity": "blockwise",
        "block_shape": (tile, tile),
        "scale_dtype": scale_dtype,
    }


MXFP8 = _scale("blockwise", 32, torch.float8_e8m0fnu)

# (granularity, block_size) groups, reused across tests.
GRANULARITIES = [
    ("tensorwise", 0),
    ("rowwise", 0),
    ("blockwise", 16),
    ("blockwise", 32),
    ("blockwise", 64),
    ("blockwise", 128),
]
# the three canonical shapes a kernel sees: no width, per-row, and a real block
CANONICAL_GRANULARITIES = [("tensorwise", 0), ("rowwise", 0), ("blockwise", 32)]
# the ragged tests use tiny groups, so their block must be small enough to tile one
RAGGED_GRANULARITIES = [("tensorwise", 0), ("rowwise", 0), ("blockwise", 4)]
# granularities that already carry a scale per row, so a ragged row axis is a no-op
PER_ROW_GRANULARITIES = [("rowwise", 0), ("blockwise", 4)]

# 100 is not a multiple of any block width here, so it leaves a partial last block
SHAPES = [(8, 256), (4, 100), (64, 128)]
# 70x130 is unaligned on both axes; the 3D entry is an expert slab
TILE_SHAPES = [(64, 128), (70, 130), (4, 64, 128)]


def _offs(counts):
    return torch.tensor(counts).cumsum(0).to(torch.int32)


def _roundtrip(x, contract_dim, fmt, scale_cfg):
    """dequant(quant(x)) — the reconstruction the quantized GEMMs actually see."""
    xq, scale = quantize_operand(x, contract_dim, fmt, scale_cfg)
    return dequantize_operand(xq, scale, contract_dim, scale_cfg)


def _rel(got, ref):
    return (got.float() - ref.float()).norm() / ref.float().norm()


def _block_amax(x, contract_dim, block):
    """Per-group amax along `contract_dim`, tiled with plain tensor ops.

    Deliberately not built from `_tile`/`_scale_block_map`: this is the independent
    side of the comparison, so it must not share the helpers under test. amax is an
    exact reduction, so equality against the implementation is bitwise.
    """
    v = x if contract_dim == -1 else x.mT
    K = v.shape[-1]
    width = block or K
    n_blocks = (K + width - 1) // width
    pad = n_blocks * width - K
    if pad:
        v = torch.nn.functional.pad(v, (0, pad))
    amax = v.abs().reshape(*v.shape[:-1], n_blocks, width).amax(-1)
    return amax if contract_dim == -1 else amax.mT


# --- _scale_block_map ---

BLOCK_MAP_CASES = [
    # (counts, block_size, row_blocks, n_blocks)
    # block_size 0 — what row/tensorwise normalize to — is one block per group
    ([3, 4, 3], 0, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2], 3),
    # bs=4 with group sizes 3,4,3: every group starts a fresh block, so none spans a
    # boundary even though 3 and 4 are not multiples of each other
    ([3, 4, 3], 4, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2], None),
    # counts 5,0,4 with bs=2 -> group 0 gets 3 blocks, group 1 none, group 2 two
    ([5, 0, 4], 2, [0, 0, 1, 1, 2, 3, 3, 4, 4], 9 // 2 + 3),
]

# (counts, block_size) shapes the boundary invariants must hold for
BLOCK_MAP_SHAPES = [
    ([5, 0, 130, 41, 1], 32),
    ([300, 5, 120, 0, 44, 210], 128),
    ([1, 1, 1], 16),
    ([3, 4, 3], 4),
    ([3, 4, 3], 0),
]


@pytest.mark.parametrize("counts,block,row_blocks,n_blocks", BLOCK_MAP_CASES)
def test_scale_block_map(counts, block, row_blocks, n_blocks):
    got_rows, got_n = _scale_block_map(_offs(counts), sum(counts), block)
    assert got_rows.tolist() == row_blocks
    if n_blocks is not None:
        assert got_n == n_blocks


@pytest.mark.parametrize("counts,block", BLOCK_MAP_SHAPES)
def test_scale_block_map_respects_group_boundaries(counts, block):
    offs, n_rows = _offs(counts), sum(counts)
    row_blocks, n_blocks = _scale_block_map(offs, n_rows, block)
    group = torch.searchsorted(
        offs, torch.arange(n_rows, dtype=torch.int32), right=True
    )
    # a block id must map to exactly one group
    for block_id in row_blocks.unique().tolist():
        assert group[row_blocks == block_id].unique().numel() == 1
    assert int(row_blocks.max()) < n_blocks  # n_blocks is a valid upper bound


# --- quantize_operand: errors ---

QUANTIZE_ERROR_CASES = [
    # (kwargs, exception, match)
    # a scale config with no scale_dtype must fail rather than pick a default
    (
        dict(
            x=torch.ones(2, 2),
            contract_dim=-1,
            fmt=E4M3,
            scale_cfg={"granularity": "rowwise", "block_shape": (0, 0)},
        ),
        KeyError,
        "scale_dtype",
    ),
    # contract_dim names one of the last two dims; 0 is outside what the tiling
    # helpers split, so it must raise rather than tile the wrong axis
    (
        dict(
            x=torch.randn(4, 8),
            contract_dim=0,
            fmt=E4M3,
            scale_cfg=_scale("tensorwise"),
        ),
        ValueError,
        "contract_dim must be -2 or -1",
    ),
    (
        dict(
            x=torch.randn(4, 8),
            contract_dim=-1,
            fmt=E4M3,
            scale_cfg=_scale("tensorwise"),
            offs=_offs([2, 2]),
        ),
        ValueError,
        "together",
    ),
    (
        dict(
            x=torch.randn(2, 4, 8),
            contract_dim=-2,
            fmt=E4M3,
            scale_cfg=_scale("tensorwise"),
            offs=_offs([2, 2]),
            ragged_dim=-2,
        ),
        ValueError,
        "2D operand",
    ),
    (
        dict(
            x=torch.randn(64, 128),
            contract_dim=-2,
            fmt=E4M3,
            scale_cfg=_scale2d(16),
            offs=_offs([32, 64]),
            ragged_dim=-2,
        ),
        NotImplementedError,
        "ragged",
    ),
]


@pytest.mark.parametrize("kwargs,exception,match", QUANTIZE_ERROR_CASES)
def test_quantize_operand_raise_error(kwargs, exception, match):
    with pytest.raises(exception, match=match):
        quantize_operand(**kwargs)


# --- quantize_operand: scale shape and value ---


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("granularity,block", GRANULARITIES)
def test_quantize_operand_scale_shape(granularity, block, shape, contract_dim):
    """The canonical layout: codes keep the operand's shape, and the scale keeps it
    too with the contracted axis replaced by its block count. (1, B) is exactly what
    the old scalar block_size produced, and a partial last block still counts as one.
    """
    x = torch.randn(*shape)
    xq, scale = quantize_operand(x, contract_dim, E4M3, _scale(granularity, block))

    K = x.shape[contract_dim]
    n_blocks = 1 if block == 0 else (K + block - 1) // block
    expected = list(x.shape)
    expected[contract_dim] = n_blocks

    assert xq.shape == x.shape and xq.dtype == torch.float8_e4m3fn
    assert tuple(scale.shape) == tuple(expected)
    assert scale.dtype == torch.float32 and scale.is_contiguous()


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block", [0, 32, 40])  # 0 is rowwise; 40 leaves a short block
def test_quantize_operand_scale_value(block, contract_dim):
    torch.manual_seed(0)
    x = torch.randn(64, 96) if contract_dim == -1 else torch.randn(96, 64)
    granularity = "rowwise" if block == 0 else "blockwise"
    _, scale = quantize_operand(x, contract_dim, E4M3, _scale(granularity, block))
    assert torch.equal(scale, _block_amax(x, contract_dim, block) / str_to_qmax(E4M3))


@pytest.mark.parametrize("granularity", ["tensorwise", "rowwise"])
def test_quantize_operand_fp32_scale_never_clips(granularity):
    # scale = amax/qmax lands the extremal element exactly on qmax, so the clamp in
    # quantize_operand is a no-op and the largest magnitude round-trips intact.
    torch.manual_seed(0)
    x = torch.randn(32, 64) * 10.0
    fq = _roundtrip(x, -1, E4M3, _scale(granularity))
    assert fq.abs().amax().item() == pytest.approx(x.abs().amax().item(), rel=1e-6)


@pytest.mark.parametrize("granularity,block", CANONICAL_GRANULARITIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_quantize_operand_compute_dtypes(granularity, block, dtype):
    x = torch.randn(8, 64, dtype=dtype)
    xq, scale = quantize_operand(x, -1, E4M3, _scale(granularity, block))
    assert xq.shape == x.shape and xq.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32  # scale is fp32 regardless of input dtype


@pytest.mark.parametrize("granularity,block", CANONICAL_GRANULARITIES)
def test_quantize_operand_zero_input(granularity, block):
    fq = _roundtrip(torch.zeros(4, 64), -1, E4M3, _scale(granularity, block))
    assert torch.isfinite(fq).all()


ROUNDTRIP_CASES = [
    # (scale_cfg, relative error bound)
    (_scale("tensorwise"), 0.1),  # e4m3 per-tensor dynamic range
    (_scale("rowwise"), 0.1),
    (_scale("blockwise", 32), 0.1),
    (MXFP8, 0.15),  # power-of-two scales give up to a binade of headroom
    (_scale2d(32), 0.15),
]


@pytest.mark.parametrize("scale_cfg,bound", ROUNDTRIP_CASES)
def test_quantize_operand_roundtrip_error(scale_cfg, bound):
    torch.manual_seed(0)
    x = torch.randn(64, 128) * 10.0
    fq = _roundtrip(x, -1, E4M3, scale_cfg)
    assert fq.shape == x.shape
    assert _rel(fq, x) < bound


# --- quantize_operand: stochastic rounding ---


@pytest.mark.parametrize("fmt", ROUNDING_FORMATS)
def test_quantize_operand_stochastic_rounding_is_unbiased(fmt):
    # The leading 1 sets the row scale; .3 is strictly between two codes for every
    # format here, so its expected reconstruction is exactly .3.
    x = torch.cat([torch.ones(1), torch.full((9999,), 0.3)]).reshape(1, -1)
    codes, scale = quantize_operand(
        x, -1, fmt, _scale("rowwise"), stochastic_rounding=True
    )
    reconstructed = codes.float() * scale

    assert codes.float()[0, 1:].unique().numel() == 2  # only the two adjacent codes
    assert reconstructed[0, 1:].mean().item() == pytest.approx(0.3, abs=0.002)


@pytest.mark.parametrize("fmt", ROUNDING_FORMATS)
def test_quantize_operand_stochastic_rounding_preserves_endpoints(fmt):
    x = torch.tensor([[1.0, 0.0, -0.0, -1.0]])
    codes, _ = quantize_operand(x, -1, fmt, _scale("rowwise"), stochastic_rounding=True)

    qmax = str_to_qmax(fmt)
    assert torch.equal(codes.float(), torch.tensor([[qmax, 0.0, -0.0, -qmax]]))
    if fmt.startswith("fp8"):
        assert torch.signbit(codes.float()[0, 2])  # negative zero survives


@pytest.mark.parametrize("fmt", FP8_FORMATS)
def test_quantize_operand_stochastic_rounding_stays_on_adjacent_codes(fmt):
    """The gate on the per-element ULP: a wrong grid lands on a non-adjacent code.

    Probes every code, every midpoint between codes, and the subnormal run below the
    smallest normal, which shares one spacing instead of halving per binade.
    """
    codes = torch.arange(256, dtype=torch.uint8).view(str_to_dtype(fmt)).float()
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


def test_quantize_operand_rounding_consumes_no_rng():
    torch.manual_seed(0)
    before = torch.random.get_rng_state()
    quantize_operand(torch.tensor([[1.0, 0.3]]), -1, E4M3, _scale("rowwise"))
    assert torch.equal(torch.random.get_rng_state(), before)


# --- quantize_operand: power-of-two (E8M0 / MX-style) scales ---


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
def test_quantize_operand_e8m0_scale_is_a_power_of_two(contract_dim):
    torch.manual_seed(0)
    x = torch.randn(64, 96) if contract_dim == -1 else torch.randn(96, 64)
    _, scale = quantize_operand(x, contract_dim, E4M3, MXFP8)
    log2_scale = torch.log2(scale[scale > 0])
    assert torch.equal(log2_scale, log2_scale.round())


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
def test_quantize_operand_e8m0_scale_lands_amax_in_the_top_binade(contract_dim):
    # 2**ceil(log2(amax/qmax)) puts amax/scale in (qmax/2, qmax]: never above, so no
    # code clips, and never a full binade below, so no range is wasted.
    torch.manual_seed(0)
    x = (torch.randn(64, 96) if contract_dim == -1 else torch.randn(96, 64)) * 7.0
    _, scale = quantize_operand(x, contract_dim, E4M3, MXFP8)
    qmax = str_to_qmax(E4M3)
    ratio = _block_amax(x, contract_dim, 32) / scale
    assert (ratio <= qmax).all() and (ratio > qmax / 2).all()


E8M0_MIN_SCALE_CASES = [
    # (fmt, the block's single nonzero value, the code it must land on)
    ("fp8_e4m3", 2**-136, 2**-9),
    ("fp8_e5m2", 2**-143, 2**-16),
    ("fp8_e4m3", 0.0, 0.0),  # an all-zero block takes the same floor
    ("fp8_e5m2", 0.0, 0.0),
]


@pytest.mark.parametrize("fmt,value,code", E8M0_MIN_SCALE_CASES)
def test_quantize_operand_e8m0_minimum_scale(fmt, value, code):
    x = torch.zeros(1, 32)
    x[0, 0] = value
    xq, scale = quantize_operand(x, -1, fmt, MXFP8)

    assert torch.equal(scale, torch.full_like(scale, 2**-127))
    assert xq.float()[0, 0] == code
    assert torch.count_nonzero(xq) == (1 if value else 0)
    assert torch.equal(dequantize_operand(xq, scale, -1, MXFP8), x)


def test_quantize_operand_e8m0_block_maximum():
    # The exponent is picked against qmax, not emax, so amax/scale <= qmax always:
    # 1.9 -> ceil(log2(1.9/448)) = -7 -> 1.9 / 2**-7 = 243.2, well inside e4m3, which
    # rounds to the nearest code (240) and comes back as 240 * 2**-7 = 1.875.
    x = torch.full((1, 32), 1.9)
    xq, scale = quantize_operand(x, -1, E4M3, MXFP8)
    assert xq.float().abs().amax().item() < str_to_qmax(E4M3)
    fq = dequantize_operand(xq, scale, -1, MXFP8)
    assert torch.allclose(fq, torch.full_like(fq, 1.875))

    # With no clipping the peak of each block loses only e4m3 round-to-nearest:
    # 3 mantissa bits, so at most half an ulp = 2**-4 relative.
    torch.manual_seed(0)
    x = (torch.randn(64, 256) * 3.0).reshape(64, 8, 32)
    fq = _roundtrip(x.reshape(64, 256), -1, E4M3, MXFP8).reshape(64, 8, 32)
    idx = x.abs().argmax(-1, keepdim=True)
    peak, got = x.gather(-1, idx).abs(), fq.gather(-1, idx).abs()
    assert ((peak - got) / peak).max().item() <= 2**-4


# --- quantize_operand: int formats ---


@pytest.mark.parametrize("granularity,block", CANONICAL_GRANULARITIES)
@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_quantize_operand_int_range(fmt, granularity, block):
    torch.manual_seed(0)
    qmax = str_to_qmax(fmt)
    x = torch.randn(64, 64) * 10.0
    codes, scale = quantize_operand(x, -1, fmt, _scale(granularity, block))

    assert codes.dtype == torch.int8  # every width is stored in int8
    assert int(codes.min()) >= -qmax and int(codes.max()) <= qmax
    if granularity == "tensorwise":
        # a tensorwise scale broadcasts one value = amax / qmax over the rows
        assert torch.allclose(scale, x.abs().max().float() / qmax)


@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_quantize_operand_int_roundtrip(fmt):
    # a uniform grid of step `scale`, so round-to-nearest is within half a step
    torch.manual_seed(0)
    x = torch.randn(128, 128)
    recon = _roundtrip(x, -1, fmt, _scale("tensorwise"))
    scale = x.abs().max().float() / str_to_qmax(fmt)
    assert (recon - x).abs().max() <= scale.item() / 2 + 1e-6


# --- quantize_operand: stacked experts ---


@pytest.mark.parametrize("granularity,block", CANONICAL_GRANULARITIES)
def test_quantize_operand_stacked_experts(granularity, block):
    # One rank-agnostic call on an (E,K,N) slab must agree with quantizing expert by
    # expert — the scale layout for a 3D operand is only exercised here.
    torch.manual_seed(0)
    b = torch.randn(3, 96, 32)
    scale_cfg = _scale(granularity, block)
    codes, scale = quantize_operand(b, -2, E4M3, scale_cfg)
    per_expert = [quantize_operand(b[g], -2, E4M3, scale_cfg) for g in range(3)]

    assert torch.equal(
        codes.view(torch.uint8),
        torch.stack([q for q, _ in per_expert]).view(torch.uint8),
    )
    assert torch.equal(scale, torch.stack([s for _, s in per_expert]))


# --- quantize_operand: ragged contraction axis (the MoE wgrad) ---

RAGGED_COUNTS = [6, 0, 5]


def _ragged_input(counts=RAGGED_COUNTS, cols=8, hot=50.0):
    """Groups differing enough in scale that a shared amax would be visible."""
    torch.manual_seed(0)
    x = torch.randn(sum(counts), cols)
    x[: counts[0]] *= hot
    return x, _offs(counts)


@pytest.mark.parametrize("granularity,block", RAGGED_GRANULARITIES)
def test_quantize_operand_ragged_scale_shape(granularity, block):
    x, offs = _ragged_input()
    _, n_blocks = _scale_block_map(offs, x.shape[0], block)
    codes, scale = quantize_operand(
        x, -2, E4M3, _scale(granularity, block), offs=offs, ragged_dim=-2
    )
    assert codes.shape == x.shape
    assert scale.shape == (n_blocks, x.shape[1])
    assert scale.dtype == torch.float32
    # the empty group still owns a row, and it must not be zero or NaN
    assert torch.isfinite(scale).all() and (scale > 0).all()


@pytest.mark.parametrize("granularity,block", RAGGED_GRANULARITIES)
def test_quantize_operand_ragged_matches_per_group(granularity, block):
    """Each group quantized on its own must give byte-identical codes AND identical
    scales to the fused ragged call — that IS the definition of expert independence.
    """
    x, offs = _ragged_input()
    scale_cfg = _scale(granularity, block)
    codes, scale = quantize_operand(x, -2, E4M3, scale_cfg, offs=offs, ragged_dim=-2)
    row_blocks, _ = _scale_block_map(offs, x.shape[0], block)

    lo = 0
    for hi in offs.tolist():
        if hi > lo:  # a single group is a non-ragged tensor: the existing code path
            expected_codes, expected_scale = quantize_operand(
                x[lo:hi], -2, E4M3, scale_cfg
            )
            assert torch.equal(codes[lo:hi], expected_codes)
            assert torch.equal(scale[row_blocks[lo:hi].unique()], expected_scale)
        lo = hi


def test_quantize_operand_ragged_isolates_group_scales():
    # The bug this whole path exists to fix: group 0 is 100x hotter than group 1.
    # A shared amax would scale both by group 0's, wiping out group 1's codes.
    x = torch.cat([torch.full((4, 3), 100.0), torch.full((4, 3), 1.0)])
    _, scale = quantize_operand(
        x, -2, E4M3, _scale("rowwise"), offs=_offs([4, 4]), ragged_dim=-2
    )
    assert torch.allclose(scale[0] / scale[1], torch.full((3,), 100.0), rtol=1e-5)


@pytest.mark.parametrize("granularity,block", RAGGED_GRANULARITIES)
def test_quantize_operand_ragged_transpose_equivalence(granularity, block):
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
    offs, scale_cfg = _offs([8, 0, 10, 6]), _scale(granularity, block)
    codes, scale = quantize_operand(x, -2, E4M3, scale_cfg, offs=offs, ragged_dim=-2)
    codes_t, scale_t = quantize_operand(
        x.mT, -1, E4M3, scale_cfg, offs=offs, ragged_dim=-1
    )
    assert torch.equal(
        codes_t.view(torch.uint8), codes.mT.contiguous().view(torch.uint8)
    )
    assert torch.equal(scale_t, scale.mT.contiguous())


# --- quantize_operand: ragged row axis (fwd/dgrad) ---


def test_quantize_operand_ragged_row_axis_is_per_group_for_tensorwise():
    x = torch.cat([torch.full((4, 3), 100.0), torch.full((4, 3), 1.0)])
    codes, scale = quantize_operand(
        x, -1, E4M3, _scale("tensorwise"), offs=_offs([4, 4]), ragged_dim=-2
    )
    assert scale.shape == (8, 1)  # canonical A layout, unchanged
    assert torch.allclose(scale[:4], scale[0].expand(4, 1))
    assert torch.allclose(scale[0] / scale[4], torch.tensor([100.0]), rtol=1e-5)
    # the cold group keeps its full code range instead of collapsing to ~1/100
    assert codes[4:].float().abs().max() > 0.5 * codes[:4].float().abs().max()


@pytest.mark.parametrize("granularity,block", PER_ROW_GRANULARITIES)
def test_quantize_operand_ragged_row_axis_is_a_noop_per_row(granularity, block):
    torch.manual_seed(0)
    x = torch.randn(8, 8)
    scale_cfg = _scale(granularity, block)
    with_ragged = quantize_operand(
        x, -1, E4M3, scale_cfg, offs=_offs([4, 4]), ragged_dim=-2
    )
    without = quantize_operand(x, -1, E4M3, scale_cfg)
    assert torch.equal(with_ragged[0], without[0])
    assert torch.equal(with_ragged[1], without[1])


# --- quantize_operand: 2D (square-tile) blockwise ---


@pytest.mark.parametrize("shape", [(64, 128), (70, 130)])
@pytest.mark.parametrize("tile", TILES)
def test_quantize_operand_blockwise_2d_is_transpose_invariant(tile, shape):
    """The property the whole feature exists for: a square tile quantizes W and W.T
    to the same codes, so the forward and dgrad see the same weight."""
    w = torch.randn(*shape)
    codes_a, _ = quantize_operand(w, -1, E4M3, _scale2d(tile))
    codes_b, _ = quantize_operand(w.t().contiguous(), -2, E4M3, _scale2d(tile))
    assert torch.equal(codes_a.t().float(), codes_b.float())


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("tile", TILES)
def test_quantize_operand_blockwise_2d_scale_layout(tile, contract_dim):
    """No kernel may need to know the tile was square: the scale leaves in the same
    shape a 1D blockwise scale of the same contract extent would."""
    x = torch.randn(64, 128)
    _, scale_2d = quantize_operand(x, contract_dim, E4M3, _scale2d(tile))
    _, scale_1d = quantize_operand(x, contract_dim, E4M3, _scale("blockwise", tile))
    assert scale_2d.shape == scale_1d.shape


@pytest.mark.parametrize("tile", TILES)
def test_quantize_operand_blockwise_2d_scale_is_tile_constant(tile):
    """Guards the expansion direction -- the one orientation-dependent step."""
    x = torch.randn(64, 128)
    _, scale = quantize_operand(x, -1, E4M3, _scale2d(tile))  # (64, 128/tile)
    for i in range(64 // tile):
        block = scale[i * tile : (i + 1) * tile, :]
        assert torch.equal(block, block[:1].expand_as(block))


@pytest.mark.parametrize("shape", TILE_SHAPES)
def test_quantize_operand_blockwise_2d_roundtrip(shape):
    x = torch.randn(*shape)
    deq = _roundtrip(x, -1, E4M3, _scale2d(32))
    assert deq.shape == x.shape
    assert (deq - x).abs().max() < 0.1 * x.abs().max()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="2D quantizer compilation requires CUDA"
)
@pytest.mark.parametrize("stochastic_rounding", [False, True])
def test_quantize_operand_compiles_fullgraph(stochastic_rounding):
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    fn = torch.compile(
        lambda: quantize_operand(
            x, -1, E4M3, _scale2d(32), stochastic_rounding=stochastic_rounding
        ),
        fullgraph=True,
    )
    _, scale = fn()
    assert scale.shape == (256, 16)


# --- dequantize_operand ---


def test_dequantize_operand_raise_error():
    # contract_dim names one of the last two dims, so 0 is outside what the tiling
    # helpers split. It must raise, not tile the wrong axis at a plausible shape.
    codes = torch.ones(3, 3, 4, dtype=torch.int8)
    scale = torch.arange(1.0, 13.0).reshape(1, 3, 4)
    with pytest.raises(ValueError, match="dim must be -2 or -1"):
        dequantize_operand(codes, scale, 0, _scale("tensorwise"))


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("granularity,block", CANONICAL_GRANULARITIES)
def test_dequantize_operand(granularity, block, contract_dim):
    torch.manual_seed(0)
    x = torch.randn(64, 96)
    scale_cfg = _scale(granularity, block)
    codes, scale = quantize_operand(x, contract_dim, E4M3, scale_cfg)
    deq = dequantize_operand(codes, scale, contract_dim, scale_cfg)
    assert deq.shape == x.shape and deq.is_contiguous()
    # metrics compare in fp32 regardless of the operand's compute dtype
    assert deq.dtype == torch.float32
    assert (
        dequantize_operand(
            *quantize_operand(x.bfloat16(), contract_dim, E4M3, scale_cfg),
            contract_dim,
            scale_cfg,
        ).dtype
        == torch.float32
    )


def test_dequantize_operand_partial_block():
    """K=100 with bs=32 gives nkb=4; a block size inferred as ceil(100/4)=25 would
    diverge, so dequantize must recover the true width from the config."""
    torch.manual_seed(0)
    x = torch.randn(4, 100)
    scale_cfg = _scale("blockwise", 32)
    codes, scale = quantize_operand(x, -1, E4M3, scale_cfg)
    deq = dequantize_operand(codes, scale, -1, scale_cfg)

    # reference built with the *true* block size, never inferred from nkb/K
    qf, sf = codes.float(), scale.float()
    pad = sf.shape[-1] * 32 - qf.shape[-1]
    qp = torch.nn.functional.pad(qf, (0, pad)) if pad else qf
    reference = (qp.unflatten(-1, (sf.shape[-1], 32)) * sf.unsqueeze(-1)).flatten(-2)[
        ..., :100
    ]
    assert torch.allclose(deq, reference, atol=1e-6)


@pytest.mark.parametrize("granularity,block", CANONICAL_GRANULARITIES)
def test_dequantize_operand_stacked_experts(granularity, block):
    # The MoE path quantizes an (E,K,N) slab along dim -2 and inverts it in one call.
    # Assert that agrees with dequantizing slab by slab — the two paths are written
    # differently, and the per-expert list is also what the deleted loop produced.
    torch.manual_seed(0)
    experts = torch.randn(4, 96, 32)
    scale_cfg = _scale(granularity, block)
    per_expert = [
        quantize_operand(experts[g], -2, E4M3, scale_cfg)
        for g in range(experts.shape[0])
    ]
    deq = dequantize_operand(
        torch.stack([q for q, _ in per_expert]),
        torch.stack([s for _, s in per_expert]),
        -2,
        scale_cfg,
    )
    reference = torch.stack(
        [
            _roundtrip(experts[g], -2, E4M3, scale_cfg).float()
            for g in range(experts.shape[0])
        ]
    )
    assert torch.equal(deq, reference)


@pytest.mark.parametrize("granularity,block", RAGGED_GRANULARITIES)
def test_dequantize_operand_ragged(granularity, block):
    x, offs = _ragged_input()
    scale_cfg = _scale(granularity, block)
    codes, scale = quantize_operand(x, -2, E4M3, scale_cfg, offs=offs, ragged_dim=-2)
    deq = dequantize_operand(codes, scale, -2, scale_cfg, offs=offs, ragged_dim=-2)

    assert deq.shape == x.shape and deq.dtype == torch.float32
    # per group, because a shared reduction would hide a mis-scaled cold group
    lo = 0
    for hi in offs.tolist():
        if hi > lo:
            assert _rel(deq[lo:hi], x[lo:hi]) < 0.1, (lo, hi)
        lo = hi


# --- _tile2d / _untile2d ---


@pytest.mark.parametrize("shape", TILE_SHAPES)
@pytest.mark.parametrize("tile", TILES)
def test_tile2d(shape, tile):
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


@pytest.mark.parametrize("tile", TILES)
def test_tile2d_amax(tile):
    x = torch.randn(64, 128)
    amax = _tile2d(x, tile).abs().amax((-3, -1))
    assert amax.shape == (64 // tile, 128 // tile)
    # a tile's amax is the max over exactly that square
    assert torch.equal(amax[1, 2], x[tile : 2 * tile, 2 * tile : 3 * tile].abs().amax())
