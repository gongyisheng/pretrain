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
    MXFP8_FORMAT_CASES,
    MXFP8_SCALE_CASES,
    MXFP8_SCALE_ERROR_CASES,
    OUT_DTYPE_CASES,
    QUANT_FORMAT_CASES,
    QUANT_SCALE_CASES,
    QUANT_SCALE_ERROR_CASES,
    SCALED_MM_CASES,
    SCALED_GROUPED_MM_CASES,
    make_grouped_mm_inputs,
    make_scaled_grouped_mm_inputs,
    make_scaled_mm_inputs,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton MM kernels are CUDA only"
)


@pytest.mark.parametrize("case", GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES, ids=lambda case: case.name)
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


@pytest.mark.parametrize("case", GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "layout", GROUPED_ERROR_LAYOUTS, ids=lambda layout: layout.replace("_", "-")
)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_grouped_mm_raise_error(case, layout, with_bias):
    """Raise error on not implemented features"""
    if layout == "ragged_n" and not with_bias:
        pytest.skip("accepted, not an error -- see test_grouped_mm_precision")
    a, b, offs, bias = make_grouped_mm_inputs(case, layout, with_bias=with_bias)
    with pytest.raises(Exception):
        triton_mm.grouped_mm(a, b, offs, bias)


@pytest.mark.parametrize("case", SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", QUANT_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES, ids=lambda case: case.name)
def test_scaled_mm_precision(case, format, scale, with_bias, out_dtype):
    capability = torch.cuda.get_device_capability()
    if format.a_format == "int8" and capability < (8, 0):
        pytest.skip("Triton INT8 scaled MM requires SM80+")
    if format.a_format.startswith("fp8") and capability < (8, 9):
        pytest.skip("Triton FP8 scaled MM requires SM89+")

    aq, bq, sa, sb, _, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, with_bias=with_bias, out_dtype=out_dtype.dtype
    )
    actual = triton_mm.scaled_mm(aq, bq, sa, sb, out_dtype.dtype, block_size, bias)
    expected = eager_mm.scaled_mm(aq, bq, sa, sb, out_dtype.dtype, block_size, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)


@pytest.mark.parametrize("case", SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", QUANT_SCALE_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_scaled_mm_raise_error(case, format, scale, with_bias):
    """Raise error on wrong scale"""
    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, with_bias=with_bias
    )
    with pytest.raises(Exception):
        triton_mm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)


@pytest.mark.parametrize("case", SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES, ids=lambda case: case.name)
def test_mxfp8_scaled_mm_precision(case, format, scale, with_bias, out_dtype):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("mxfp8 scaled MM requires SM100+")

    aq, bq, sa, sb, _, block_size, bias = make_scaled_mm_inputs(
        case,
        format,
        scale,
        scale_dtype=torch.float8_e8m0fnu,
        with_bias=with_bias,
        out_dtype=out_dtype.dtype,
    )
    actual = triton_mm.mxfp8_scaled_mm(
        aq, bq, sa, sb, out_dtype.dtype, block_size, bias
    )
    expected = eager_mm.scaled_mm(aq, bq, sa, sb, out_dtype.dtype, block_size, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-4, rel


@pytest.mark.parametrize("case", SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_mxfp8_scaled_mm_raise_error(case, format, scale, with_bias):
    """Raise error on wrong scale"""
    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, with_bias=with_bias
    )
    with pytest.raises(Exception):
        triton_mm.mxfp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)


# ---------------------------------------------------------------------------
# Scaled grouped MM: fp8/int8 ragged expert matmul
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", QUANT_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES, ids=lambda case: case.name)
def test_scaled_grouped_mm_precision(case, scale, format, layout, with_bias, out_dtype):
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

    aq, bq, sa, sb, offs, _, kbs, bias = make_scaled_grouped_mm_inputs(
        case, layout, format, scale, with_bias=with_bias, out_dtype=out_dtype.dtype
    )
    actual = triton_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype.dtype, kbs, bias=bias
    )
    expected = eager_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype.dtype, kbs, bias=bias
    )
    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", QUANT_SCALE_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_scaled_grouped_mm_raise_error_on_scale(case, layout, format, scale, with_bias):
    """The scale-block widths scaled_grouped_mm declines, over every ragged layout."""
    if layout == "ragged_n" and not with_bias:
        pytest.skip("accepted, not an error -- see test_scaled_grouped_mm_precision")
    aq, bq, sa, sb, offs, out_dtype, kbs, bias = make_scaled_grouped_mm_inputs(
        case, layout, format, scale, with_bias=with_bias
    )
    with pytest.raises(Exception):
        triton_mm.scaled_grouped_mm(aq, bq, sa, sb, offs, out_dtype, kbs, bias=bias)


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "layout", GROUPED_ERROR_LAYOUTS, ids=lambda layout: layout.replace("_", "-")
)
@pytest.mark.parametrize("format", QUANT_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", QUANT_SCALE_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_scaled_grouped_mm_raise_error_on_layout(
    case, layout, format, scale, with_bias
):
    """Raise error on bad layout"""
    if layout == "ragged_n" and not with_bias:
        pytest.skip("accepted, not an error -- see test_scaled_grouped_mm_precision")
    aq, bq, sa, sb, offs, out_dtype, kbs, bias = make_scaled_grouped_mm_inputs(
        case, layout, format, scale, with_bias=with_bias
    )
    with pytest.raises(Exception):
        triton_mm.scaled_grouped_mm(aq, bq, sa, sb, offs, out_dtype, kbs, bias=bias)


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES, ids=lambda case: case.name)
def test_mxfp8_scaled_grouped_mm_precision(
    case, layout, format, scale, with_bias, out_dtype
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

    aq, bq, sa, sb, offs, _, kbs, bias = make_scaled_grouped_mm_inputs(
        case,
        layout,
        format,
        scale,
        with_bias=with_bias,
        scale_dtype=torch.float8_e8m0fnu,
        out_dtype=out_dtype.dtype,
    )
    actual = triton_mm.mxfp8_scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype.dtype, kbs, bias=bias
    )
    expected = eager_mm.scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype.dtype, kbs, bias=bias
    )

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-4, rel


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_ERROR_CASES, ids=lambda case: case.name)
def test_mxfp8_scaled_grouped_mm_raise_error(case, layout, format, scale):
    aq, bq, sa, sb, offs, out_dtype, block_size, bias = make_scaled_grouped_mm_inputs(
        case, layout, format, scale, with_bias=False
    )
    with pytest.raises(Exception):
        triton_mm.mxfp8_scaled_grouped_mm(
            aq, bq, sa, sb, offs, out_dtype, block_size, bias=bias
        )
