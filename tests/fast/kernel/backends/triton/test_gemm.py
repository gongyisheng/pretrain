"""Triton GEMM parity against the eager correctness backend."""

import pytest
import torch

from src.kernel.backends.eager import gemm as eager_gemm
from src.kernel.backends.triton import gemm as triton_gemm
from src.quant.quantize import quantize_operand
from tests.fast.kernel.backends.helper import (
    BIAS_CASES,
    DENSE_GEMM_SHAPES,
    GemmShape,
    GROUPED_LAYOUTS,
    GROUPED_GEMM_SHAPES,
    QuantFormatCase,
    QuantScaleCase,
    QUANT_FORMAT_CASES,
    QUANT_SCALE_CASES,
    make_grouped_mm_inputs,
    make_scaled_grouped_mm_inputs,
    make_scaled_mm_inputs,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton GEMM kernels are CUDA only"
)

E4M3 = torch.float8_e4m3fn
FMT = {E4M3: "fp8_e4m3"}
FP8_E4M3_FORMAT = QuantFormatCase("fp8_e4m3", "fp8_e4m3", "fp8_e4m3", 0, 0)
INT8_FORMAT = QuantFormatCase("int8", "int8", "int8", 0, 0)
TENSORWISE_SCALING = QuantScaleCase("tensorwise", "tensorwise", (0, 0), 0)
ROWWISE_SCALING = QuantScaleCase("rowwise", "rowwise", (1, 0), 0)
BLOCK_16_SCALING = QuantScaleCase("1x16", "blockwise", (1, 16), 16)
BLOCK_32_SCALING = QuantScaleCase("1x32", "blockwise", (1, 32), 32)
BLOCK_48_SCALING = QuantScaleCase("1x48", "blockwise", (1, 48), 48)
BLOCK_64_SCALING = QuantScaleCase("1x64", "blockwise", (1, 64), 64)
BLOCK_128_SCALING = QuantScaleCase("1x128", "blockwise", (1, 128), 128)


def _scaling(gran, bs=0, scale_dtype=torch.float32):
    return {
        "granularity": gran,
        "block_shape": (1, bs) if bs else (0, 0),
        "scale_dtype": scale_dtype,
    }


def _run(a, b, gran, bs, a_fmt, b_fmt):
    # the kernel takes the 0-sentinel width; the oracle's pad math takes a count
    scaling = _scaling(gran, bs)
    aq, sa = quantize_operand(a, -1, a_fmt, scaling)
    bq, sb = quantize_operand(b, -2, b_fmt, scaling)
    out = triton_gemm.scaled_mm(aq, bq, sa, sb, torch.float32, bs)
    oracle = eager_gemm.scaled_mm(
        aq,
        bq,
        sa,
        sb,
        torch.float32,
        bs or a.shape[1],
    )
    return out, oracle


def _make(counts, K, N, seed=0):
    """Build (a (R,K), b=w.mT (E,K,N) transposed view, offs (E,) int32) on CUDA/bf16."""
    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    b = (torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.1).mT
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return a, b, offs


# The three ragged layouts, named by which dim `offs` partitions. Counts sum to a
# multiple of 8 so every operand satisfies torch's 16-byte stride alignment; the
# empty group exercises the degenerate case.
LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")
_LAYOUT_COUNTS = [64, 0, 130, 46]  # sums to 240


def _make_layout(layout, counts=_LAYOUT_COUNTS, M=32, K=64, N=48, seed=0):
    """Build operands for one grouped ragged layout."""
    torch.manual_seed(seed)
    G, R = len(counts), sum(counts)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)

    def rand(dim0, dim1, dim2=None):
        shape = (dim0, dim1) if dim2 is None else (dim0, dim1, dim2)
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1

    if layout == "ragged_m":  # (R,K) x (G,K,N) -> (R,N)
        return rand(R, K), rand(G, N, K).mT, offs
    if layout == "ragged_k":  # (M,R) x (R,N) -> (G,M,N)
        return rand(R, M).mT, rand(R, N), offs
    if layout == "ragged_n":  # (G,M,K) x (K,R) -> (M,R)
        return rand(G, M, K), rand(K, R), offs
    raise ValueError(layout)


@pytest.mark.parametrize("shape", GROUPED_GEMM_SHAPES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.bfloat16, 2e-2), (torch.float16, 2e-3)],
    ids=["bf16", "fp16"],
)
def test_grouped_mm_precision(shape, layout, with_bias, dtype, tolerance):
    if layout == "ragged_n" and with_bias:
        pytest.skip("ragged-N has no defined per-expert output-channel bias")
    a, b, offs, bias = make_grouped_mm_inputs(
        shape, layout, with_bias=with_bias, dtype=dtype
    )

    actual = triton_gemm.grouped_mm(a, b, offs, bias)
    expected = eager_gemm.grouped_mm(a, b, offs, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "gemm_fn",
    [triton_gemm.scaled_mm, triton_gemm.mxfp8_scaled_mm],
    ids=["scaled", "mxfp8"],
)
@pytest.mark.parametrize("shape", DENSE_GEMM_SHAPES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scaling", QUANT_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize(
    "out_dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"]
)
def test_scaled_mm_precision(shape, format, scaling, with_bias, gemm_fn, out_dtype):
    capability = torch.cuda.get_device_capability()
    if format.a_format == "int8" and capability < (8, 0):
        pytest.skip("Triton INT8 scaled GEMM requires SM80+")
    if format.a_format.startswith("fp8") and capability < (8, 9):
        pytest.skip("Triton FP8 scaled GEMM requires SM89+")

    is_mxfp8 = gemm_fn is triton_gemm.mxfp8_scaled_mm
    if is_mxfp8:
        if format.a_format == "int8":
            pytest.skip("mxfp8 kernel has no integer format")
        bs = scaling.block_size
        if bs == 0 or bs % 32 != 0:
            pytest.skip("mxfp8 kernel requires block_size a nonzero multiple of 32")
        if capability < (10, 0):
            pytest.skip("mxfp8 scaled GEMM requires SM100+")

    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        shape,
        format,
        scaling,
        scale_dtype=torch.float8_e8m0fnu if is_mxfp8 else torch.float32,
        with_bias=with_bias,
        dtype=out_dtype,
    )
    actual = gemm_fn(aq, bq, sa, sb, out_dtype, block_size, bias)
    expected = eager_gemm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)


def test_output_stride_is_last_dim_dense_and_row_aligned():
    a, b, offs = _make_layout("ragged_m", N=51)
    got = triton_gemm.grouped_mm(a, b, offs)
    eager = eager_gemm.grouped_mm(a, b, offs)
    alignment = 16 // got.element_size()
    expected_row_stride = -(-got.shape[-1] // alignment) * alignment
    assert got.stride(-1) == 1
    assert got.stride(-2) == expected_row_stride
    torch.testing.assert_close(got.float(), eager.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_grouped_mm_empty_and_uneven_groups(layout):
    a, b, offs = _make_layout(layout, counts=[5, 0, 130, 41, 1])
    got = triton_gemm.grouped_mm(a, b, offs)
    expected = eager_gemm.grouped_mm(a, b, offs)
    torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "K,N",
    [
        (64, 384),  # latent-MoE gate_up
        (192, 64),  # latent-MoE down
    ],
)
def test_grouped_mm_latent_moe_shapes(K, N):
    """Latent-MoE forward and wgrad shapes with skewed expert loads."""
    counts = [300, 5, 120, 0, 44, 210, 7, 90, 33, 150, 1, 260, 12, 80, 40, 400]
    a, b, offs = _make(counts, K=K, N=N, seed=1)
    got = triton_gemm.grouped_mm(a, b, offs)
    torch.testing.assert_close(
        got.float(), eager_gemm.grouped_mm(a, b, offs).float(), rtol=2e-2, atol=2e-2
    )
    R = sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    grad_c = torch.randn(R, N, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = eager_gemm.grouped_mm(a.mT, grad_c, offs)
    got = triton_gemm.grouped_mm(a.mT, grad_c, offs)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_nonmultiple_shapes_mask_correctly():
    torch.manual_seed(0)
    a = torch.randn(70, 100, device="cuda", dtype=torch.bfloat16)  # M,K odd
    b = torch.randn(100, 130, device="cuda", dtype=torch.bfloat16)  # N odd, K%32!=0
    out, oracle = _run(a, b, "blockwise", 32, FMT[E4M3], FMT[E4M3])
    torch.testing.assert_close(out, oracle, rtol=3e-2, atol=3e-2)


def test_scaled_mm_float32_output_dtype_is_respected():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], _scaling("rowwise"))
    bq, sb = quantize_operand(b, -2, FMT[E4M3], _scaling("rowwise"))
    out = triton_gemm.scaled_mm(aq, bq, sa, sb, torch.float32, 0)
    expected = eager_gemm.scaled_mm(aq, bq, sa, sb, torch.float32, 0)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Scaled GEMM: the mxfp8 path (tl.dot_scaled / QMMA.SF)
# ---------------------------------------------------------------------------

MX = _scaling("blockwise", 32, torch.float8_e8m0fnu)


@pytest.mark.parametrize("shape", [(256, 512, 192), (128, 256, 96), (192, 160, 128)])
def test_scaled_mm_mxfp8_adds_no_error_over_dequantizing(shape):
    """Applying the e8m0 scales inside the MMA must be as exact as dequantizing first.

    The bound is fp32-rounding tight (measured 2e-8..4e-8) and pins the instruction's
    arithmetic rather than merely its quantization-level parity.
    """
    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], MX)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], MX)
    got = triton_gemm.mxfp8_scaled_mm(aq, bq, sa, sb, torch.float32, 32)
    oracle = eager_gemm.scaled_mm(aq, bq, sa, sb, torch.float32, 32)
    rel = (got - oracle).norm() / oracle.norm()
    assert rel < 1e-6, rel


@pytest.mark.parametrize(
    ("granularity", "block_size"),
    [("tensorwise", 0), ("blockwise", 16)],
    ids=["zero-sentinel", "narrow-block"],
)
def test_mxfp8_scaled_mm_declines_invalid_block_size(granularity, block_size):
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 96, device="cuda", dtype=torch.bfloat16)
    scaling = _scaling(granularity, block_size, torch.float8_e8m0fnu)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], scaling)
    with pytest.raises(ValueError, match="nonzero multiple of 32"):
        triton_gemm.mxfp8_scaled_mm(aq, bq, sa, sb, torch.float32, block_size)


def rand(rows, columns):
    return torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("bs", [32, 64, 128])
def test_mxfp8_kernel_matches_scaled_mm_when_k_is_not_a_multiple_of_32(bs):
    """K not a multiple of 32 leaves sa/sb with a trailing partial scale block --
    n_scale_blocks is ceil(K/32), one more than K // 32. The bs=32 case is the one that
    pins the regression: block_size == _MXFP8_BLOCK_SIZE is supposed to be a bit-exact
    no-op, so if this fails only at bs=32 the bug is in the block count, not the
    replication.
    """
    torch.manual_seed(0)
    a, b = rand(64, 163), rand(163, 96)
    scaling = _scaling("blockwise", bs, torch.float8_e8m0fnu)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], scaling)
    bq, sb = quantize_operand(b, -2, FMT[E4M3], scaling)
    mx = triton_gemm.mxfp8_scaled_mm(aq, bq, sa, sb, torch.float32, bs)
    scaled = triton_gemm.scaled_mm(aq, bq, sa, sb, torch.float32, bs)
    torch.testing.assert_close(mx, scaled, rtol=2e-2, atol=2e-2)


_VM, _VK, _VN = 64, 256, 96


def test_scaled_mm_declines_non_power_of_two_multi_block_size():
    a = torch.randn(_VM, _VK, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(_VK, _VN, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], _scaling("blockwise", 48))
    bq, sb = quantize_operand(b, -2, FMT[E4M3], _scaling("blockwise", 48))
    with pytest.raises(ValueError, match="power of two >= 16"):
        triton_gemm.scaled_mm(aq, bq, sa, sb, torch.float32, 48)


def test_scaled_mm_accepts_non_pow2_block_size_spanning_k():
    """Guards the check above against over-rejecting: a block_size >= K collapses to
    a single scale block, which no BLOCK_K ever has to tile, so it needs no power of
    two.
    """
    a = torch.randn(_VM, _VK, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(_VK, _VN, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, FMT[E4M3], _scaling("blockwise", _VK + 44))
    bq, sb = quantize_operand(b, -2, FMT[E4M3], _scaling("blockwise", _VK + 44))
    assert sa.shape[1] == 1
    torch.testing.assert_close(
        triton_gemm.scaled_mm(aq, bq, sa, sb, torch.float32, _VK + 44),
        eager_gemm.scaled_mm(aq, bq, sa, sb, torch.float32, _VK + 44),
        rtol=2e-2,
        atol=2e-2,
    )


# ---------------------------------------------------------------------------
# Scaled grouped GEMM: fp8/int8 ragged expert matmul
# ---------------------------------------------------------------------------


SCALED_LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")
_SCALED_COUNTS = [64, 0, 130, 46]  # sums to 240; the empty group is deliberate
SCALED_GROUPED_SHAPE = GemmShape("scaled-grouped", 64, 48)
# The ragged-contraction layout dequants the same codes as its oracle and only
# reorders the fp32 accumulation, so it must agree far more tightly than the
# forward layouts, whose oracle pads scale blocks independently of the tiling.
_SCALED_TOL = {"ragged_m": 0.02, "ragged_k": 1e-5, "ragged_n": 0.02}


def _expected_shape(layout, counts, M=32, N=48):
    E, R = len(counts), sum(counts)
    return {"ragged_m": (R, N), "ragged_k": (E, M, N), "ragged_n": (M, R)}[layout]


@pytest.mark.parametrize(
    "scaling",
    # 48 is deliberately not a multiple of any BLOCK_K: the kernel masks the
    # reduction to the scale block's end rather than clamping BLOCK_K to it, so
    # correctness must not depend on which config autotune picks.
    [
        ROWWISE_SCALING,
        TENSORWISE_SCALING,
        BLOCK_16_SCALING,
        BLOCK_32_SCALING,
        BLOCK_48_SCALING,
    ],
    ids=lambda case: case.name,
)
@pytest.mark.parametrize(
    "format", [FP8_E4M3_FORMAT, INT8_FORMAT], ids=lambda case: case.name
)
@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
@pytest.mark.parametrize(
    "out_dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"]
)
def test_scaled_grouped_layouts_match_oracle(layout, format, scaling, out_dtype):
    aq, bq, sa, sb, offs, kbs = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_SHAPE,
        layout,
        format,
        scaling,
        batch_size=1,
        counts=_SCALED_COUNTS,
    )
    got = triton_gemm.scaled_grouped_mm(aq, bq, sa, sb, offs, out_dtype, kbs)
    ref = eager_gemm.scaled_grouped_mm(aq, bq, sa, sb, offs, out_dtype, kbs)
    assert got.shape == _expected_shape(layout, _SCALED_COUNTS)
    assert got.dtype == out_dtype
    assert torch.isfinite(got).all()
    got_float, ref_float = got.float(), ref.float()
    rel = (got_float - ref_float).norm() / ref_float.norm().clamp_min(1e-12)
    tolerance = (
        max(_SCALED_TOL[layout], 2e-3)
        if out_dtype is torch.float16
        else _SCALED_TOL[layout]
    )
    assert rel < tolerance, rel


@pytest.mark.parametrize(
    "scaling", [BLOCK_32_SCALING, BLOCK_64_SCALING, BLOCK_128_SCALING]
)
def test_scaled_grouped_mxfp8_ragged_m_matches_oracle_when_k_is_not_a_multiple_of_32(
    scaling,
):
    """Same regression as the dense-gemm K=163 case, for the ragged-M grouped layout --
    K here is shared by every group (not ragged), so it is the layout the generalized
    condition applies to. A dropped trailing scale column here does not just lose
    accuracy: the kernel's own `scale_block_end = tl.cdiv(K, 32)` still expects it, so
    reading past a host-side tensor narrowed one column short produces NaN, not just a
    large relative error.
    """
    aq, bq, sa, sb, offs, kbs = make_scaled_grouped_mm_inputs(
        GemmShape("scaled-grouped-tail", 163, 48),
        "ragged_m",
        FP8_E4M3_FORMAT,
        scaling,
        batch_size=1,
        counts=_SCALED_COUNTS,
        scale_dtype=torch.float8_e8m0fnu,
    )
    got = triton_gemm.mxfp8_scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    ref = eager_gemm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    assert torch.isfinite(got).all(), got
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < _SCALED_TOL["ragged_m"], rel


@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_mxfp8_adds_no_error_over_dequantizing(layout):
    """As for the dense kernel: fp32-rounding tight, not quantization-noise loose.

    Per-layout tolerances here are 1e-5 (ragged_k) to 2e-2 (ragged_m/n), so this is
    what would catch the group cursor drifting a scale block -- which stays well inside
    2e-2 while being plainly wrong.
    """
    aq, bq, sa, sb, offs, kbs = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_SHAPE,
        layout,
        FP8_E4M3_FORMAT,
        BLOCK_32_SCALING,
        batch_size=1,
        counts=_SCALED_COUNTS,
        scale_dtype=torch.float8_e8m0fnu,
    )
    got = triton_gemm.mxfp8_scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    oracle = eager_gemm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    assert got.shape == _expected_shape(layout, _SCALED_COUNTS)
    assert torch.isfinite(got).all()
    rel = (got.float() - oracle).norm() / oracle.norm().clamp_min(1e-12)
    assert rel < 1e-6, rel


@pytest.mark.parametrize(
    ("layout", "scaling"),
    [("ragged_m", TENSORWISE_SCALING), ("ragged_k", BLOCK_64_SCALING)],
    ids=["zero-sentinel", "ragged-k-wide-block"],
)
def test_scaled_grouped_mxfp8_declines_invalid_block_size(layout, scaling):
    aq, bq, sa, sb, offs, kbs = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_SHAPE,
        layout,
        FP8_E4M3_FORMAT,
        scaling,
        batch_size=1,
        counts=_SCALED_COUNTS,
        scale_dtype=torch.float8_e8m0fnu,
    )
    with pytest.raises(ValueError):
        triton_gemm.mxfp8_scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)


def test_scaled_grouped_ragged_k_empty_group_is_zero():
    # an expert with no tokens owns no blockwise scale blocks; its (M,N) slice of the
    # (E,M,N) output must still be zeroed rather than left uninitialized
    counts = [8, 0, 8]
    aq, bq, sa, sb, offs, kbs = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_SHAPE,
        "ragged_k",
        FP8_E4M3_FORMAT,
        BLOCK_32_SCALING,
        batch_size=1,
        counts=counts,
    )
    out = triton_gemm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    assert (out[1] == 0).all()


@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_layouts_bfloat16_output_dtype(layout):
    aq, bq, sa, sb, offs, kbs = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_SHAPE,
        layout,
        FP8_E4M3_FORMAT,
        ROWWISE_SCALING,
        batch_size=1,
        counts=_SCALED_COUNTS,
        seed=3,
    )
    out = triton_gemm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.bfloat16, kbs)
    expected = eager_gemm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.bfloat16, kbs)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "gemm_fn",
    [triton_gemm.scaled_grouped_mm, triton_gemm.mxfp8_scaled_grouped_mm],
    ids=["scaled", "mxfp8"],
)
@pytest.mark.parametrize("layout", ("ragged_m", "ragged_k"))
def test_scaled_grouped_bias_matches_oracle(layout, gemm_fn):
    """Fused (G,N) bias, broadcast over the output's row dim.

    Blockwise is the discriminating case: the bias is unscaled and belongs on the
    final accumulator, so folding it into a per-scale-block partial would both
    multiply it by that block's scales and add it once per block.
    """
    is_mxfp8 = gemm_fn is triton_gemm.mxfp8_scaled_grouped_mm
    if is_mxfp8 and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("mxfp8 scaled grouped GEMM requires SM100+")
    aq, bq, sa, sb, offs, kbs = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_SHAPE,
        layout,
        FP8_E4M3_FORMAT,
        BLOCK_32_SCALING,
        batch_size=1,
        counts=_SCALED_COUNTS,
        scale_dtype=torch.float8_e8m0fnu if is_mxfp8 else torch.float32,
    )
    bias = torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16)
    got = gemm_fn(aq, bq, sa, sb, offs, torch.float32, kbs, bias=bias)
    ref = eager_gemm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, torch.float32, kbs, bias=bias
    )
    tolerance = 1e-6 if is_mxfp8 else _SCALED_TOL[layout]
    torch.testing.assert_close(got.float(), ref.float(), rtol=tolerance, atol=tolerance)
