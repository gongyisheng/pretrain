"""cuBLASLt MM parity against the eager correctness backend."""

import pytest
import torch

import src.kernel.backends.cublaslt._C  # noqa: F401
from src.kernel.backends.cublaslt import gemm as cublaslt_mm
from src.kernel.backends.eager import gemm as eager_mm
from tests.fast.kernel.backends.helper import (
    BF16_OUT,
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

MXFP8_SUPPORT_CASES = (
    (torch.float8_e4m3fn, torch.float8_e4m3fn, 32, True),
    (torch.float8_e5m2, torch.float8_e4m3fn, 32, False),
    (torch.float8_e4m3fn, torch.float8_e5m2, 32, False),
    (torch.float8_e4m3fn, torch.float8_e4m3fn, 64, False),
)


@pytest.mark.parametrize(
    "a_dtype,b_dtype,block_size,supported",
    MXFP8_SUPPORT_CASES,
    ids=["e4m3-1x32", "e5m2-a", "e5m2-b", "block-64"],
)
def test_supports_mxfp8_scaled_mm(a_dtype, b_dtype, block_size, supported):
    """The probe must decline whatever the kernel rejects, or the selector routes
    a call here that the C++ contract then refuses instead of falling to Triton."""
    aq = torch.zeros(256, 512, device="cuda").to(a_dtype)
    bq = torch.zeros(512, 128, device="cuda").to(b_dtype)

    assert (
        cublaslt_mm.supports_mxfp8_scaled_mm(aq, bq, torch.bfloat16, block_size)
        is supported
    )


@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_mxfp8_scaled_mm_precision(case, format, scale, with_bias):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("cuBLASLt mxfp8 scaled MM requires SM100+")

    # the kernel returns bf16 and rejects every other request by contract, so
    # unlike the Triton tests there is no out_dtype axis to bound against
    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, scale_dtype=torch.float8_e8m0fnu, with_bias=with_bias
    )
    actual = cublaslt_mm.scaled_mm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias)
    expected = eager_mm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)

    assert actual.shape == expected.shape
    assert actual.dtype is BF16_OUT.dtype
    assert actual.stride() == (actual.shape[1], 1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=BF16_OUT.rtol, atol=BF16_OUT.atol)
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-4, rel


@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_ERROR_CASES, ids=lambda case: case.name)
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
    message = f"MXFP8 GEMM requires block_size=32, got {block_size}"
    with pytest.raises(ValueError, match=message):
        cublaslt_mm.scaled_mm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias)
