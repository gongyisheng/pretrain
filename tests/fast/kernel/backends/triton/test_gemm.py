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
    if format.a_format == "int8" and not cuda_capability_at_least((8, 0)):
        pytest.skip("Triton INT8 scaled MM requires SM80+")
    if format.a_format.startswith("fp8") and not cuda_capability_at_least((8, 9)):
        pytest.skip("Triton FP8 scaled MM requires SM89+")

    aq, bq, sa, sb, _, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, with_bias=with_bias, out_dtype=out_dtype.dtype
    )
    actual = triton_mm.scaled_mm(aq, bq, sa, sb, out_dtype.dtype, block_size, bias)
    expected = eager_mm.scaled_mm(aq, bq, sa, sb, out_dtype.dtype, block_size, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )


@pytest.mark.parametrize("backend", GLOBAL_SCALE_BACKENDS, ids=["triton", "eager"])
@pytest.mark.parametrize("case", SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "format", GLOBAL_SCALE_FORMAT_CASES, ids=lambda case: case.name
)
@pytest.mark.parametrize("scale", QUANT_SCALE_CASES, ids=lambda case: case.name)
def test_scaled_mm_global_scale_matches_folded_scale(backend, case, format, scale):
    """A global scale in the epilogue equals folding it into the block scales.

    The factors are powers of two, so folding them into fp32 block scales is exact
    and both orderings produce bit-identical fp32 output. That makes this an oracle
    for the epilogue's semantics, which triton-vs-eager parity alone cannot supply:
    both backends could apply the scale wrongly in the same way.
    """
    aq, bq, sa, sb, _, block_size, _ = make_scaled_mm_inputs(
        case, format, scale, with_bias=False, out_dtype=torch.float32
    )
    ga = torch.full((1,), 4.0, device=aq.device)
    gb = torch.full((1,), 2.0, device=aq.device)
    got = backend.scaled_mm(aq, bq, sa, sb, torch.float32, block_size, None, ga, gb)
    want = backend.scaled_mm(
        aq, bq, sa * 4.0, sb * 2.0, torch.float32, block_size, None
    )
    assert torch.equal(got, want)


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


@cuda_sm100_or_newer
@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", MXFP8_PLUS_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_PLUS_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES, ids=lambda case: case.name)
def test_mxfp8_scaled_mm_precision(case, format, scale, with_bias, out_dtype):
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
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-4, rel


@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", MXFP8_PLUS_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "scale", MXFP8_PLUS_SCALE_ERROR_CASES, ids=lambda case: case.name
)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_mxfp8_scaled_mm_raise_error(case, format, scale, with_bias):
    """Raise error on wrong scale"""
    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case,
        format,
        scale,
        with_bias=with_bias,
        scale_dtype=torch.float8_e8m0fnu,
    )
    message = f"mxfp8 block_size must be a nonzero multiple of 32, got {block_size}"
    with pytest.raises(ValueError, match=message):
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
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )


@pytest.mark.parametrize("backend", GLOBAL_SCALE_BACKENDS, ids=["triton", "eager"])
@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize(
    "format", GLOBAL_SCALE_FORMAT_CASES, ids=lambda case: case.name
)
@pytest.mark.parametrize("scale", QUANT_SCALE_CASES, ids=lambda case: case.name)
def test_scaled_grouped_mm_global_scale_matches_folded_scale(
    backend, case, layout, format, scale
):
    """Each group's global scale equals folding that group's factor into its scales.

    Distinct per-group factors are what separate a correct per-group epilogue from
    one that reuses group 0's value for every group.
    """
    if layout in {"ragged_k", "ragged_n"} and all(
        dimension > 1 for dimension in scale.block_shape
    ):
        pytest.skip("2D scales cross ragged groups during quantization")
    aq, bq, sa, sb, offs, _, kbs, _ = make_scaled_grouped_mm_inputs(
        case, layout, format, scale, with_bias=False, out_dtype=torch.float32
    )
    groups = offs.numel()
    # Distinct powers of two per group: exact under fp32, and a product of 4**g that
    # no group shares, so reusing one group's factor everywhere fails the assert.
    exponents = torch.arange(groups, device=aq.device, dtype=torch.float32)
    ga, gb = 2.0**exponents, 2.0**exponents
    got = backend.scaled_grouped_mm(
        aq, bq, sa, sb, offs, torch.float32, kbs, None, ga, gb
    )
    base = backend.scaled_grouped_mm(aq, bq, sa, sb, offs, torch.float32, kbs, None)

    factor = ga * gb
    want = base.clone()
    if layout == "ragged_k":  # (E, M, N): one output slab per group
        want *= factor[:, None, None]
    else:
        ends = offs.tolist()
        for group, (lo, hi) in enumerate(zip([0, *ends[:-1]], ends)):
            if layout == "ragged_m":  # (M, N): output rows partitioned by offs
                want[lo:hi] *= factor[group]
            else:  # ragged_n -> (M, N): output columns partitioned by offs
                want[:, lo:hi] *= factor[group]
    assert torch.equal(got, want)


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


@cuda_sm100_or_newer
@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", MXFP8_PLUS_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_PLUS_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
@pytest.mark.parametrize("out_dtype", OUT_DTYPE_CASES, ids=lambda case: case.name)
def test_mxfp8_scaled_grouped_mm_precision(
    case, layout, format, scale, with_bias, out_dtype
):
    """The grouped analogue of test_mxfp8_scaled_mm_precision."""
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
    torch.testing.assert_close(
        actual, expected, rtol=out_dtype.rtol, atol=out_dtype.atol
    )
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 5e-6, rel


@pytest.mark.parametrize("case", SCALED_GROUPED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("format", MXFP8_PLUS_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "scale", MXFP8_PLUS_SCALE_ERROR_CASES, ids=lambda case: case.name
)
def test_mxfp8_scaled_grouped_mm_raise_error(case, layout, format, scale):
    aq, bq, sa, sb, offs, out_dtype, block_size, bias = make_scaled_grouped_mm_inputs(
        case,
        layout,
        format,
        scale,
        with_bias=False,
        scale_dtype=torch.float8_e8m0fnu,
    )
    message = (
        "mxfp8 grouped GEMM requires block_size a nonzero multiple "
        f"of 32, got {block_size}"
    )
    with pytest.raises(ValueError, match=message):
        triton_mm.mxfp8_scaled_grouped_mm(
            aq, bq, sa, sb, offs, out_dtype, block_size, bias=bias
        )
