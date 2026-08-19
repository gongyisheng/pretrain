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
    MXFP8_SCALED_MM_ERROR_CASES,
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


@pytest.mark.parametrize(
    "case", MXFP8_SCALED_MM_ERROR_CASES, ids=lambda case: case.name
)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale", MXFP8_SCALE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_mxfp8_scaled_mm_raise_error_on_shape(case, format, scale, with_bias):
    """Raise error on a contraction the 16-byte operand alignment cannot express"""
    aq, bq, sa, sb, out_dtype, block_size, bias = make_scaled_mm_inputs(
        case, format, scale, with_bias=with_bias
    )
    with pytest.raises(Exception):
        cublaslt_mm.scaled_mm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("ok", True),
        ("block_size_64", False),
        ("out_fp16", False),
        # M=250 tokens leaves a 250-byte row stride on the transposed wgrad operand
        ("unaligned_tokens", False),
    ],
)
def test_supports_mxfp8_scaled_mm_matches_the_kernel_contract(
    case: str, expected: bool
):
    """The routing predicate must accept exactly what the kernel accepts.

    A disagreement would either lose cuBLASLt on valid calls or route invalid ones
    into a raise.
    """
    k = 250 if case == "unaligned_tokens" else 256
    aq = torch.zeros(256, k, device="cuda", dtype=torch.float8_e4m3fn)
    bq = torch.zeros(k, 128, device="cuda", dtype=torch.float8_e4m3fn)
    block_size = 64 if case == "block_size_64" else 32
    out_dtype = torch.float16 if case == "out_fp16" else torch.bfloat16

    assert (
        cublaslt_mm.supports_mxfp8_scaled_mm(aq, bq, out_dtype, block_size) is expected
    )


@pytest.mark.parametrize(
    ("shape", "with_bias"),
    [
        ((0, 16, 16), False),
        ((16, 16, 0), False),
        ((16, 0, 16), False),
        ((16, 0, 16), True),
    ],
)
def test_mxfp8_scaled_mm_zero_size_matches_eager(
    shape: tuple[int, int, int], with_bias: bool
):
    """An empty expert must still produce the eager result rather than raise."""
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("cuBLASLt mxfp8 scaled MM requires SM100+")

    m, k, n = shape
    blocks = -(-k // 32)
    aq = torch.empty((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    bq = torch.empty((k, n), device="cuda", dtype=torch.float8_e4m3fn)
    sa = torch.ones((m, blocks), device="cuda")
    sb = torch.ones((blocks, n), device="cuda")
    bias = (
        torch.linspace(-1, 1, n, device="cuda", dtype=torch.bfloat16)
        if with_bias
        else None
    )

    actual = cublaslt_mm.scaled_mm_mxfp8(aq, bq, sa, sb, torch.bfloat16, 32, bias)
    expected = eager_mm.scaled_mm(aq, bq, sa, sb, torch.bfloat16, 32, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
