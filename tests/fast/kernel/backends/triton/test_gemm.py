"""Triton MM parity against the eager correctness backend."""

import pytest
import torch

from src.kernel.backends.eager import gemm as eager_mm
from src.kernel.backends.triton import gemm as triton_mm
from tests.fast.kernel.backends.helper import (
    BIAS_CASES,
    GROUPED_LAYOUTS,
    GROUPED_ERROR_LAYOUTS,
    GROUPED_MM_CASES,
    MXFP8_SCALED_MM_CASES,
    MXFP8_PLUS_FORMAT_CASES,
    MXFP8_PLUS_SCALE_CASES,
    MXFP8_PLUS_SCALE_ERROR_CASES,
    MXFP8_SCALED_GROUPED_MM_CASES,
    OUT_DTYPE_CASES,
    NVFP4_SCALED_GROUPED_MM_CASES,
    NVFP4_SCALED_MM_CASES,
    NVFP4_BLOCK_SIZES,
    NVFP4_SCALE_DTYPES,
    QUANT_FORMAT_CASES,
    QUANT_SCALE_CASES,
    QUANT_SCALE_ERROR_CASES,
    SCALED_MM_CASES,
    SCALED_GROUPED_MM_CASES,
    make_grouped_mm_inputs,
    make_scaled_grouped_mm_inputs,
    make_scaled_mm_inputs,
)
from tests.fast.helper import (
    cuda_capability_at_least,
    cuda_only,
    cuda_sm100_or_newer,
)


pytestmark = cuda_only

# The global-scale epilogue is a backend contract, so both implement it.
GLOBAL_SCALE_BACKENDS = (triton_mm, eager_mm)
# The epilogue runs after the dot, so it cannot depend on the element format. One
# integer and one fp8 pair cover both operand-dtype paths into it. All five formats
# were run once and agreed, at 1040 cells against 416; the rest are compile-time
# specialisations of a path the element format does not reach.
GLOBAL_SCALE_FORMAT_CASES = tuple(
    case for case in QUANT_FORMAT_CASES if case.name in {"int8xint8", "e4m3xe4m3"}
)


@pytest.mark.parametrize("case", GROUPED_MM_CASES)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES)
def test_grouped_mm_precision(case, layout, with_bias, out_dtype):
    if layout == "ragged_n" and with_bias:
        pytest.skip("rejected, not supported -- see test_grouped_mm_raise_error")
    a, b, offs, bias = make_grouped_mm_inputs(
        case, layout, with_bias=with_bias, out_dtype=out_dtype.dtype
    )

    actual = triton_mm.grouped_mm(a, b, offs, bias)
    expected = eager_mm.grouped_mm(a, b, offs, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    # the output must itself be a legal operand when backward feeds it back
    alignment = 16 // actual.element_size()
    assert actual.stride(-1) == 1
    assert actual.stride(-2) == -(-actual.shape[-1] // alignment) * alignment
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )


@pytest.mark.parametrize("case", GROUPED_MM_CASES)
@pytest.mark.parametrize("layout", GROUPED_ERROR_LAYOUTS)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
def test_grouped_mm_raise_error(case, layout, with_bias):
    """Raise error on not implemented features"""
    if layout == "ragged_n" and not with_bias:
        pytest.skip("accepted, not an error -- see test_grouped_mm_precision")
    a, b, offs, bias = make_grouped_mm_inputs(case, layout, with_bias=with_bias)
    with pytest.raises(Exception):
        triton_mm.grouped_mm(a, b, offs, bias)


@pytest.mark.parametrize("case", SCALED_MM_CASES)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES)
@pytest.mark.parametrize("scale", QUANT_SCALE_CASES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES)
@pytest.mark.parametrize("with_global_scale", (False, True))
def test_scaled_mm_precision(
    case, format, scale, with_bias, out_dtype, with_global_scale
):
    if format.a_format == "int8" and not cuda_capability_at_least((8, 0)):
        pytest.skip("Triton INT8 scaled MM requires SM80+")
    if format.a_format.startswith("fp8") and not cuda_capability_at_least((8, 9)):
        pytest.skip("Triton FP8 scaled MM requires SM89+")

    aq, bq, sa, sb, _, block_size, bias, gsa, gsb = make_scaled_mm_inputs(
        case,
        format,
        scale,
        with_bias=with_bias,
        out_dtype=out_dtype.dtype,
        with_global_scale=with_global_scale,
    )
    actual = triton_mm.scaled_mm(
        aq, bq, sa, sb, out_dtype.dtype, block_size, bias, gsa, gsb
    )
    expected = eager_mm.scaled_mm(
        aq, bq, sa, sb, out_dtype.dtype, block_size, bias, gsa, gsb
    )

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )


@pytest.mark.parametrize("case", SCALED_MM_CASES)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES)
@pytest.mark.parametrize("scale", QUANT_SCALE_ERROR_CASES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
def test_scaled_mm_raise_error(case, format, scale, with_bias):
    """Raise error on wrong scale"""
    aq, bq, sa, sb, out_dtype, block_size, bias, _, _ = make_scaled_mm_inputs(
        case, format, scale, with_bias=with_bias
    )
    with pytest.raises(Exception):
        triton_mm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)


@cuda_sm100_or_newer
@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES)
@pytest.mark.parametrize("format", MXFP8_PLUS_FORMAT_CASES)
@pytest.mark.parametrize("scale", MXFP8_PLUS_SCALE_CASES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES)
@pytest.mark.parametrize("with_global_scale", (False, True))
def test_mxfp8_scaled_mm_precision(
    case, format, scale, with_bias, out_dtype, with_global_scale
):
    aq, bq, sa, sb, _, block_size, bias, gsa, gsb = make_scaled_mm_inputs(
        case,
        format,
        scale,
        scale_dtype=torch.float8_e8m0fnu,
        with_bias=with_bias,
        out_dtype=out_dtype.dtype,
        with_global_scale=with_global_scale,
    )
    actual = triton_mm.mxfp8_scaled_mm(
        aq, bq, sa, sb, out_dtype.dtype, block_size, bias, gsa, gsb
    )
    expected = eager_mm.scaled_mm(
        aq, bq, sa, sb, out_dtype.dtype, block_size, bias, gsa, gsb
    )

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-4, rel


@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES)
@pytest.mark.parametrize("format", MXFP8_PLUS_FORMAT_CASES)
@pytest.mark.parametrize("scale", MXFP8_PLUS_SCALE_ERROR_CASES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
def test_mxfp8_scaled_mm_raise_error(case, format, scale, with_bias):
    """Raise error on wrong scale"""
    aq, bq, sa, sb, out_dtype, block_size, bias, _, _ = make_scaled_mm_inputs(
        case,
        format,
        scale,
        with_bias=with_bias,
        scale_dtype=torch.float8_e8m0fnu,
    )
    message = f"mxfp8 block_size must be a nonzero multiple of 32, got {block_size}"
    with pytest.raises(ValueError, match=message):
        triton_mm.mxfp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)


@cuda_sm100_or_newer
@pytest.mark.parametrize("case", NVFP4_SCALED_MM_CASES)
@pytest.mark.parametrize("block_size", NVFP4_BLOCK_SIZES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES)
@pytest.mark.parametrize("with_global_scale", (False, True))
@pytest.mark.parametrize("scale_dtype", NVFP4_SCALE_DTYPES)
def test_nvfp4_scaled_mm_precision(
    case, block_size, with_bias, out_dtype, with_global_scale, scale_dtype
):
    aq, bq, sa, sb, dtype, block_size, bias, gsa, gsb = make_scaled_mm_inputs(
        case,
        with_bias=with_bias,
        out_dtype=out_dtype.dtype,
        with_global_scale=with_global_scale,
        packed_e2m1=True,
        block_size=block_size,
    )
    sa = sa.to(scale_dtype)
    sb = sb.to(scale_dtype)
    actual = triton_mm.nvfp4_scaled_mm(
        aq, bq, sa, sb, dtype, block_size, bias, gsa, gsb
    )
    expected = eager_mm.scaled_mm(
        aq, bq, sa, sb, dtype, block_size, bias, gsa, gsb, unpack_e2m1=True
    )

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )


@cuda_sm100_or_newer
@pytest.mark.parametrize("case", NVFP4_SCALED_MM_CASES)
@pytest.mark.parametrize("block_size", NVFP4_BLOCK_SIZES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("scale_arg", ("sa", "sb"))
def test_nvfp4_scaled_mm_raise_error(case, block_size, with_bias, scale_arg):
    aq, bq, sa, sb, dtype, block_size, bias, _, _ = make_scaled_mm_inputs(
        case,
        with_bias=with_bias,
        packed_e2m1=True,
        block_size=block_size,
    )
    if scale_arg == "sa":
        sa = sa.to(torch.float8_e8m0fnu)
    else:
        sb = sb.to(torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="MXFP4"):
        triton_mm.nvfp4_scaled_mm(aq, bq, sa, sb, dtype, block_size, bias)


# ---------------------------------------------------------------------------
# Scaled grouped MM: fp8/int8 ragged expert matmul
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES)
@pytest.mark.parametrize("scale", QUANT_SCALE_CASES)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES)
@pytest.mark.parametrize("with_global_scale", (False, True))
def test_scaled_grouped_mm_precision(
    case, scale, format, layout, with_bias, out_dtype, with_global_scale
):
    if format.a_format == "int8" and not cuda_capability_at_least((8, 0)):
        pytest.skip("Triton INT8 scaled grouped MM requires SM80+")
    if format.a_format.startswith("fp8") and not cuda_capability_at_least((8, 9)):
        pytest.skip("Triton FP8 scaled grouped MM requires SM89+")
    if layout == "ragged_n" and with_bias:
        pytest.skip("rejected, not supported -- see test_scaled_grouped_mm_raise_error")
    if layout in {"ragged_k", "ragged_n"} and all(
        dimension > 1 for dimension in scale.block_shape
    ):
        pytest.skip("2D scales cross ragged groups during quantization")

    aq, bq, sa, sb, offs, _, kbs, bias, gsa, gsb = make_scaled_grouped_mm_inputs(
        case,
        layout,
        format,
        scale,
        with_bias=with_bias,
        out_dtype=out_dtype.dtype,
        with_global_scale=with_global_scale,
    )
    actual = triton_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype.dtype, kbs, bias, gsa, gsb
    )
    expected = eager_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype.dtype, kbs, bias, gsa, gsb
    )
    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES)
@pytest.mark.parametrize("scale", QUANT_SCALE_ERROR_CASES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
def test_scaled_grouped_mm_raise_error_on_scale(case, layout, format, scale, with_bias):
    """The scale-block widths scaled_grouped_mm declines, over every ragged layout."""
    if layout == "ragged_n" and not with_bias:
        pytest.skip("accepted, not an error -- see test_scaled_grouped_mm_precision")
    aq, bq, sa, sb, offs, out_dtype, kbs, bias, _, _ = make_scaled_grouped_mm_inputs(
        case, layout, format, scale, with_bias=with_bias
    )
    with pytest.raises(Exception):
        triton_mm.scaled_grouped_mm(aq, bq, sa, sb, offs, out_dtype, kbs, bias=bias)


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES)
@pytest.mark.parametrize("layout", GROUPED_ERROR_LAYOUTS)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES)
@pytest.mark.parametrize("scale", QUANT_SCALE_ERROR_CASES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
def test_scaled_grouped_mm_raise_error_on_layout(
    case, layout, format, scale, with_bias
):
    """Raise error on bad layout"""
    if layout == "ragged_n" and not with_bias:
        pytest.skip("accepted, not an error -- see test_scaled_grouped_mm_precision")
    aq, bq, sa, sb, offs, out_dtype, kbs, bias, _, _ = make_scaled_grouped_mm_inputs(
        case, layout, format, scale, with_bias=with_bias
    )
    with pytest.raises(Exception):
        triton_mm.scaled_grouped_mm(aq, bq, sa, sb, offs, out_dtype, kbs, bias=bias)


@cuda_sm100_or_newer
@pytest.mark.parametrize("case", MXFP8_SCALED_GROUPED_MM_CASES)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", MXFP8_PLUS_FORMAT_CASES)
@pytest.mark.parametrize("scale", MXFP8_PLUS_SCALE_CASES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES)
@pytest.mark.parametrize("with_global_scale", (False, True))
def test_mxfp8_scaled_grouped_mm_precision(
    case, layout, format, scale, with_bias, out_dtype, with_global_scale
):
    """The grouped analogue of test_mxfp8_scaled_mm_precision."""
    if layout == "ragged_n" and with_bias:
        pytest.skip("rejected, not supported -- see test_scaled_grouped_mm_raise_error")
    if layout in {"ragged_k", "ragged_n"} and all(
        dimension > 1 for dimension in scale.block_shape
    ):
        pytest.skip("2D scales cross ragged groups during quantization")

    aq, bq, sa, sb, offs, _, kbs, bias, gsa, gsb = make_scaled_grouped_mm_inputs(
        case,
        layout,
        format,
        scale,
        with_bias=with_bias,
        scale_dtype=torch.float8_e8m0fnu,
        out_dtype=out_dtype.dtype,
        with_global_scale=with_global_scale,
    )
    if with_global_scale:
        gsa = torch.arange(1, offs.numel() + 1, device=aq.device, dtype=torch.float32)
        gsb = torch.full_like(gsa, 2.0)
    actual = triton_mm.mxfp8_scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype.dtype, kbs, bias, gsa, gsb
    )
    expected = eager_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype.dtype, kbs, bias, gsa, gsb
    )

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 5e-5, rel


@pytest.mark.parametrize("case", MXFP8_SCALED_GROUPED_MM_CASES)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", MXFP8_PLUS_FORMAT_CASES)
@pytest.mark.parametrize("scale", MXFP8_PLUS_SCALE_ERROR_CASES)
def test_mxfp8_scaled_grouped_mm_raise_error(case, layout, format, scale):
    aq, bq, sa, sb, offs, out_dtype, block_size, bias, _, _ = (
        make_scaled_grouped_mm_inputs(
            case,
            layout,
            format,
            scale,
            with_bias=False,
            scale_dtype=torch.float8_e8m0fnu,
        )
    )
    message = (
        "mxfp8 grouped GEMM requires block_size a nonzero multiple "
        f"of 32, got {block_size}"
    )
    with pytest.raises(ValueError, match=message):
        triton_mm.mxfp8_scaled_grouped_mm(
            aq, bq, sa, sb, offs, out_dtype, block_size, bias=bias
        )


@cuda_sm100_or_newer
@pytest.mark.parametrize("case", NVFP4_SCALED_GROUPED_MM_CASES)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("block_size", NVFP4_BLOCK_SIZES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES)
@pytest.mark.parametrize("scale_dtype", NVFP4_SCALE_DTYPES)
def test_nvfp4_scaled_grouped_mm_precision(
    case, layout, block_size, with_bias, out_dtype, scale_dtype
):
    if layout == "ragged_n" and with_bias:
        pytest.skip("bias is not supported for ragged-N")
    aq, bq, sa, sb, offs, dtype, block_size, bias, gsa, gsb = (
        make_scaled_grouped_mm_inputs(
            case,
            layout,
            with_bias=with_bias,
            out_dtype=out_dtype.dtype,
            with_global_scale=True,
            packed_e2m1=True,
            block_size=block_size,
        )
    )
    sa = sa.to(scale_dtype)
    sb = sb.to(scale_dtype)
    actual = triton_mm.nvfp4_scaled_grouped_mm(
        aq, bq, sa, sb, offs, dtype, block_size, bias, gsa, gsb
    )
    expected = eager_mm.scaled_grouped_mm(
        aq,
        bq,
        sa,
        sb,
        offs,
        dtype,
        block_size,
        bias,
        gsa,
        gsb,
        unpack_e2m1=True,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )


@cuda_sm100_or_newer
@pytest.mark.parametrize("case", NVFP4_SCALED_GROUPED_MM_CASES)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("block_size", NVFP4_BLOCK_SIZES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("scale_arg", ("sa", "sb"))
def test_nvfp4_scaled_grouped_mm_raise_error_on_scale(
    case, layout, block_size, with_bias, scale_arg
):
    aq, bq, sa, sb, offs, dtype, block_size, bias, _, _ = make_scaled_grouped_mm_inputs(
        case,
        layout,
        with_bias=with_bias,
        out_dtype=torch.float32,
        packed_e2m1=True,
        block_size=block_size,
    )
    if scale_arg == "sa":
        sa = sa.to(torch.float8_e8m0fnu)
    else:
        sb = sb.to(torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="MXFP4"):
        triton_mm.nvfp4_scaled_grouped_mm(aq, bq, sa, sb, offs, dtype, block_size, bias)


@cuda_sm100_or_newer
@pytest.mark.parametrize("case", NVFP4_SCALED_GROUPED_MM_CASES)
@pytest.mark.parametrize("block_size", NVFP4_BLOCK_SIZES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("layout", GROUPED_ERROR_LAYOUTS)
def test_nvfp4_scaled_grouped_mm_raise_error_on_layout(
    case, block_size, layout, with_bias
):
    if layout == "ragged_n" and not with_bias:
        pytest.skip(
            "accepted, not an error -- see test_nvfp4_scaled_grouped_mm_precision"
        )
    aq, bq, sa, sb, offs, dtype, block_size, bias, _, _ = make_scaled_grouped_mm_inputs(
        case,
        layout,
        with_bias=with_bias,
        out_dtype=torch.float32,
        packed_e2m1=True,
        block_size=block_size,
    )
    with pytest.raises(NotImplementedError):
        triton_mm.nvfp4_scaled_grouped_mm(aq, bq, sa, sb, offs, dtype, block_size, bias)
