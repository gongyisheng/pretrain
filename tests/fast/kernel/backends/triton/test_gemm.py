"""Triton MM parity against the eager correctness backend."""

import pytest
import torch

from src.kernel.backends.eager import gemm as eager_mm
from src.kernel.backends.triton import gemm as triton_mm
from tests.fast.kernel.backends.helper import (
    BIAS_CASES,
    BLOCKWISE1D_32_SCALE,
    BLOCKWISE1D_64_SCALE,
    BLOCKWISE1D_128_SCALE,
    GROUPED_LAYOUTS,
    GROUPED_MM_CASES,
    GROUPED_MM_ERROR_CASES,
    GROUPED_ERROR_LAYOUTS,
    MXFP8_SCALE_ERROR_CASES,
    MXFP8_FORMAT_CASES,
    MXFP8_SCALE_CASES,
    OUT_DTYPE_CASES,
    QuantFormatCase,
    QUANT_FORMAT_CASES,
    QUANT_SCALE_CASES,
    ScaledGroupedMMCase,
    SCALED_GROUPED_MM_CASE,
    SCALED_GROUPED_MM_CASES,
    SCALED_MM_CASES,
    SCALED_MM_ERROR_CASES,
    QUANT_SCALE_ERROR_CASES,
    RAGGED_COUNTS,
    TENSORWISE_SCALE,
    make_grouped_mm_inputs,
    make_scaled_grouped_mm_inputs,
    make_scaled_mm_inputs,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton MM kernels are CUDA only"
)

FP8_E4M3_FORMAT = QuantFormatCase("fp8_e4m3", "fp8_e4m3", "fp8_e4m3", 0, 0)
INT8_FORMAT = QuantFormatCase("int8", "int8", "int8", 0, 0)


@pytest.mark.parametrize("case", GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.bfloat16, 2e-2), (torch.float16, 2e-3)],
    ids=["bf16", "fp16"],
)
def test_grouped_mm_precision(case, layout, with_bias, dtype, tolerance):
    """Every ragged layout, over shapes that stress the tile masking.

    `odd-n` leaves fixed N unaligned, while `RAGGED_COUNTS` supplies uneven expert
    loads including empty and single-row groups; ragged-N also leaves its output's
    last dimension unaligned.
    """
    if layout == "ragged_n" and with_bias:
        pytest.skip("rejected, not supported -- see test_grouped_mm_raise_error")
    a, b, offs, bias = make_grouped_mm_inputs(
        case, layout, with_bias=with_bias, dtype=dtype
    )

    actual = triton_mm.grouped_mm(a, b, offs, bias)
    expected = eager_mm.grouped_mm(a, b, offs, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    # the output must itself be a legal operand when backward feeds it back
    alignment = 16 // actual.element_size()
    assert actual.stride(-1) == 1
    assert actual.stride(-2) == -(-actual.shape[-1] // alignment) * alignment
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("case", GROUPED_MM_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "layout", GROUPED_ERROR_LAYOUTS, ids=lambda layout: layout.replace("_", "-")
)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_grouped_mm_raise_error(case, layout, with_bias):
    if layout == "ragged_n" and not with_bias:
        pytest.skip("accepted, not an error -- see test_grouped_mm_precision")
    a, b, offs, bias = make_grouped_mm_inputs(case, layout, with_bias=with_bias)
    with pytest.raises(NotImplementedError):
        triton_mm.grouped_mm(a, b, offs, bias)


@pytest.mark.parametrize("case", SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", QUANT_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES, ids=["bf16", "fp16"])
def test_scaled_mm_precision(case, format, scale, with_bias, out_dtype):
    capability = torch.cuda.get_device_capability()
    if format.a_format == "int8" and capability < (8, 0):
        pytest.skip("Triton INT8 scaled MM requires SM80+")
    if format.a_format.startswith("fp8") and capability < (8, 9):
        pytest.skip("Triton FP8 scaled MM requires SM89+")

    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, with_bias=with_bias, dtype=out_dtype
    )
    actual = triton_mm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)
    expected = eager_mm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)


@pytest.mark.parametrize("case", SCALED_MM_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", QUANT_SCALE_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_scaled_mm_raise_error(case, format, scale, with_bias):
    """The scale-block widths scaled_mm declines, whatever else the call carries.

    The guard reads only shapes, so it holds over every fp8 and int8 operand pair.
    """
    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, with_bias=with_bias
    )
    with pytest.raises(ValueError, match="power of two >= 16"):
        triton_mm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)


# ---------------------------------------------------------------------------
# Scaled MM: the mxfp8 path (tl.dot_scaled / QMMA.SF)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize(
    ("out_dtype", "global_rel_tol"),
    [(torch.bfloat16, 1e-4), (torch.float16, 1e-4)],
    ids=["bf16", "fp16"],
)
def test_mxfp8_scaled_mm_precision(
    case, format, scale, with_bias, out_dtype, global_rel_tol
):
    """The same matrix as test_scaled_mm_precision over the formats and scale widths
    the mxfp8 kernel accepts, with e8m0 scales.
    """
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("mxfp8 scaled MM requires SM100+")

    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case,
        format,
        scale,
        scale_dtype=torch.float8_e8m0fnu,
        with_bias=with_bias,
        dtype=out_dtype,
    )
    actual = triton_mm.mxfp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)
    expected = eager_mm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < global_rel_tol, rel


@pytest.mark.parametrize("case", SCALED_MM_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_mxfp8_scaled_mm_raise_error(case, scale, with_bias):
    """The scale-block widths mxfp8_scaled_mm declines, whatever else the call carries.

    The guard runs before either scale tensor is read, so float32 scales serve here
    rather than the e8m0 ones the kernel would otherwise require.
    """
    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case, FP8_E4M3_FORMAT, scale, with_bias=with_bias
    )
    with pytest.raises(ValueError, match="nonzero multiple of 32"):
        triton_mm.mxfp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)


# ---------------------------------------------------------------------------
# Scaled grouped MM: fp8/int8 ragged expert matmul
# ---------------------------------------------------------------------------


# The ragged-contraction layout dequants the same codes as its oracle and only
# reorders the fp32 accumulation, so it must agree far more tightly than the
# forward layouts, whose oracle pads scale blocks independently of the tiling.
_SCALED_TOL = {"ragged_m": 0.02, "ragged_k": 1e-5, "ragged_n": 0.02}


def _expected_shape(layout, case):
    groups, rows = len(case.counts), sum(case.counts)
    return {
        "ragged_m": (rows, case.n),
        "ragged_k": (groups, case.m, case.n),
        "ragged_n": (case.m, rows),
    }[layout]


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", QUANT_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES, ids=["bf16", "fp16"])
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_scaled_grouped_mm_precision(case, layout, format, scale, out_dtype, with_bias):
    capability = torch.cuda.get_device_capability()
    if format.a_format == "int8" and capability < (8, 0):
        pytest.skip("Triton INT8 scaled grouped MM requires SM80+")
    if format.a_format.startswith("fp8") and capability < (8, 9):
        pytest.skip("Triton FP8 scaled grouped MM requires SM89+")
    if layout == "ragged_n" and with_bias:
        pytest.skip("rejected, not supported -- see test_scaled_grouped_mm_raise_error")
    if layout in {"ragged_k", "ragged_n"} and all(
        dimension > 1 for dimension in scale.block_shape
    ):
        pytest.skip("2D scales cross ragged groups during quantization")

    aq, bq, sa, sb, offs, kbs, bias = make_scaled_grouped_mm_inputs(
        case, layout, format, scale, with_bias=with_bias
    )
    actual = triton_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype, kbs, bias=bias
    )
    expected = eager_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype, kbs, bias=bias
    )
    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize(
    ("out_dtype", "global_rel_tol"),
    [(torch.bfloat16, 1e-4), (torch.float16, 1e-4)],
    ids=["bf16", "fp16"],
)
def test_mxfp8_scaled_grouped_mm_precision(
    case, layout, format, scale, with_bias, out_dtype, global_rel_tol
):
    """The grouped analogue of test_mxfp8_scaled_mm_precision."""
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("mxfp8 scaled grouped MM requires SM100+")
    if layout == "ragged_n" and with_bias:
        pytest.skip("rejected, not supported -- see test_scaled_grouped_mm_raise_error")
    if layout in {"ragged_k", "ragged_n"} and all(
        dimension > 1 for dimension in scale.block_shape
    ):
        pytest.skip("2D scales cross ragged groups during quantization")

    aq, bq, sa, sb, offs, kbs, bias = make_scaled_grouped_mm_inputs(
        case,
        layout,
        format,
        scale,
        with_bias=with_bias,
        scale_dtype=torch.float8_e8m0fnu,
    )
    actual = triton_mm.mxfp8_scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype, kbs, bias=bias
    )
    expected = eager_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype, kbs, bias=bias
    )

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < global_rel_tol, rel


@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", QUANT_SCALE_ERROR_CASES, ids=lambda case: case.name)
def test_scaled_grouped_mm_rejects_unsupported_scale_block_width(layout, format, scale):
    aq, bq, sa, sb, offs, block_size, _ = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_MM_CASE,
        layout,
        format,
        scale,
        with_bias=False,
    )

    with pytest.raises(ValueError, match="power of two >= 16"):
        triton_mm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, block_size)


@pytest.mark.parametrize(
    ("layout", "with_bias"),
    [("3d_x_3d", False), ("ragged_n", True)],
    ids=["3d-x-3d", "ragged-n-bias"],
)
def test_scaled_grouped_mm_raise_error(layout, with_bias):
    """Reject the two operand combinations that no grouped layout defines.

    Removing either backend rejection guard lets its corresponding malformed call
    reach kernel selection instead of raising NotImplementedError.
    """
    aq, bq, sa, sb, offs, kbs, bias = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_MM_CASE,
        layout,
        INT8_FORMAT,
        BLOCKWISE1D_32_SCALE,
        with_bias=with_bias,
    )
    with pytest.raises(NotImplementedError):
        triton_mm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs, bias=bias)


@pytest.mark.parametrize(
    "scale", [BLOCKWISE1D_32_SCALE, BLOCKWISE1D_64_SCALE, BLOCKWISE1D_128_SCALE]
)
def test_scaled_grouped_mxfp8_ragged_m_matches_oracle_when_k_is_not_a_multiple_of_32(
    scale,
):
    """Same regression as the dense-mm K=163 case, for the ragged-M grouped layout --
    K here is shared by every group (not ragged), so it is the layout the generalized
    condition applies to. A dropped trailing scale column here does not just lose
    accuracy: the kernel's own `scale_block_end = tl.cdiv(K, 32)` still expects it, so
    reading past a host-side tensor narrowed one column short produces NaN, not just a
    large relative error.
    """
    aq, bq, sa, sb, offs, kbs, _ = make_scaled_grouped_mm_inputs(
        ScaledGroupedMMCase("scaled-grouped-tail", 163, 48, RAGGED_COUNTS),
        "ragged_m",
        FP8_E4M3_FORMAT,
        scale,
        with_bias=False,
        scale_dtype=torch.float8_e8m0fnu,
    )
    got = triton_mm.mxfp8_scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    ref = eager_mm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    assert torch.isfinite(got).all(), got
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < _SCALED_TOL["ragged_m"], rel


@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
def test_scaled_grouped_mxfp8_adds_no_error_over_dequantizing(layout):
    """As for the dense kernel: fp32-rounding tight, not quantization-noise loose.

    Per-layout tolerances here are 1e-5 (ragged_k) to 2e-2 (ragged_m/n), so this is
    what would catch the group cursor drifting a scale block -- which stays well inside
    2e-2 while being plainly wrong.
    """
    aq, bq, sa, sb, offs, kbs, _ = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_MM_CASE,
        layout,
        FP8_E4M3_FORMAT,
        BLOCKWISE1D_32_SCALE,
        with_bias=False,
        scale_dtype=torch.float8_e8m0fnu,
    )
    got = triton_mm.mxfp8_scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    oracle = eager_mm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    assert got.shape == _expected_shape(layout, SCALED_GROUPED_MM_CASE)
    assert torch.isfinite(got).all()
    rel = (got.float() - oracle).norm() / oracle.norm().clamp_min(1e-12)
    assert rel < 1e-6, rel


@pytest.mark.parametrize(
    ("layout", "scale"),
    [("ragged_m", TENSORWISE_SCALE), ("ragged_k", BLOCKWISE1D_64_SCALE)],
    ids=["zero-sentinel", "ragged-k-wide-block"],
)
def test_scaled_grouped_mxfp8_declines_invalid_block_size(layout, scale):
    aq, bq, sa, sb, offs, kbs, _ = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_MM_CASE,
        layout,
        FP8_E4M3_FORMAT,
        scale,
        with_bias=False,
        scale_dtype=torch.float8_e8m0fnu,
    )
    with pytest.raises(ValueError):
        triton_mm.mxfp8_scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)


def test_scaled_grouped_ragged_k_empty_group_is_zero():
    # an expert with no tokens owns no blockwise scale blocks; its (M,N) slice of the
    # (E,M,N) output must still be zeroed rather than left uninitialized
    aq, bq, sa, sb, offs, kbs, _ = make_scaled_grouped_mm_inputs(
        ScaledGroupedMMCase("empty-group", 64, 48, (8, 0, 8)),
        "ragged_k",
        FP8_E4M3_FORMAT,
        BLOCKWISE1D_32_SCALE,
        with_bias=False,
    )
    out = triton_mm.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs)
    assert (out[1] == 0).all()


@pytest.mark.parametrize("layout", ("ragged_m", "ragged_k"))
def test_scaled_grouped_mxfp8_bias_matches_oracle(layout):
    """Fused (G,N) bias, broadcast over the output's row dim.

    Blockwise is the discriminating case: the bias is unscaled and belongs on the
    final accumulator, so folding it into a per-scale-block partial would both
    multiply it by that block's scales and add it once per block.
    """
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("mxfp8 scaled grouped MM requires SM100+")
    aq, bq, sa, sb, offs, kbs, bias = make_scaled_grouped_mm_inputs(
        SCALED_GROUPED_MM_CASE,
        layout,
        FP8_E4M3_FORMAT,
        BLOCKWISE1D_32_SCALE,
        with_bias=True,
        scale_dtype=torch.float8_e8m0fnu,
    )
    got = triton_mm.mxfp8_scaled_grouped_mm(
        aq, bq, sa, sb, offs, torch.float32, kbs, bias=bias
    )
    ref = eager_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, torch.float32, kbs, bias=bias
    )
    torch.testing.assert_close(got.float(), ref.float(), rtol=1e-6, atol=1e-6)
