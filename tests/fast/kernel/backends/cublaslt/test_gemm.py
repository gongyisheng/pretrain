"""cuBLASLt MM parity against the eager correctness backend."""

import pytest
import torch

import src.kernel.backends.cublaslt._C  # noqa: F401
from src.kernel.backends.cublaslt import gemm as cublaslt_mm
from src.kernel.backends.eager import gemm as eager_mm
from tests.fast.kernel.backends.helper import (
    BIAS_CASES,
    MXFP8_FORMAT_CASES,
    MXFP8_SCALE_CASES,
    MXFP8_SCALE_ERROR_CASES,
    MXFP8_SCALED_MM_CASES,
    make_scaled_mm_inputs,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuBLASLt MM kernels are CUDA only"
)


@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_mxfp8_scaled_mm_precision(case, format, scale, with_bias):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("cuBLASLt mxfp8 scaled MM requires SM100+")

    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, scale_dtype=torch.float8_e8m0fnu, with_bias=with_bias
    )
    actual = cublaslt_mm.scaled_mm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias)
    expected = eager_mm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == out_dtype
    assert actual.stride() == (actual.shape[1], 1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=format.rtol, atol=format.atol)
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-4, rel


@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_ERROR_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_mxfp8_scaled_mm_raise_error(case, format, scale, with_bias):
    """Raise error on wrong scale"""
    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, with_bias=with_bias
    )
    with pytest.raises(Exception):
        cublaslt_mm.scaled_mm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias)
