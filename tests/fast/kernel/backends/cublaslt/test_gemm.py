import pytest
import torch

from src.kernel.backends.cublaslt import gemm as cublaslt_mm
from src.kernel.backends.eager import gemm as eager_mm
from tests.fast.kernel.backends.helper import (
    BF16_OUT,
    BIAS_CASES,
    MXFP8_FORMAT_CASES,
    MXFP8_SCALE_CASES,
    MXFP8_SCALE_ERROR_CASES,
    MXFP8_SCALED_MM_CASES,
    NVFP4_SCALED_MM_CASES,
    OUT_DTYPE_CASES,
    make_scaled_mm_inputs,
)
from tests.fast.helper import cuda_only, cuda_sm100_or_newer


MXFP8_SUPPORT_INPUTS = {
    "aq": torch.zeros((32, 32), dtype=torch.float8_e4m3fn),
    "bq": torch.zeros((32, 32), dtype=torch.float8_e4m3fn),
    "sa": torch.ones((32, 1), dtype=torch.float8_e8m0fnu),
    "sb": torch.ones((1, 32), dtype=torch.float8_e8m0fnu),
    "out_dtype": torch.bfloat16,
    "block_size": 32,
    "bias": None,
}

MXFP8_SUPPORT_CASES = (
    ({}, True),
    ({"aq": MXFP8_SUPPORT_INPUTS["aq"][0]}, False),
    ({"aq": MXFP8_SUPPORT_INPUTS["aq"].to(torch.float8_e5m2)}, False),
    ({"bq": MXFP8_SUPPORT_INPUTS["bq"].to(torch.float8_e5m2)}, False),
    ({"out_dtype": torch.float16}, False),
    ({"block_size": 64}, False),
    ({"sa": MXFP8_SUPPORT_INPUTS["sa"].to(torch.float8_e4m3fn)}, False),
    ({"sa": torch.ones((31, 1), dtype=torch.float8_e8m0fnu)}, False),
    ({"bias": torch.zeros(32, dtype=torch.float16)}, False),
    ({"bias": torch.zeros(31, dtype=torch.bfloat16)}, False),
    ({"bias": torch.zeros(64, dtype=torch.bfloat16)[::2]}, False),
)

NVFP4_SUPPORT_INPUTS = {
    "aq": torch.zeros((32, 16), dtype=torch.uint8),
    "bq": torch.zeros((16, 32), dtype=torch.uint8),
    "sa": torch.ones((32, 2), dtype=torch.float8_e4m3fn),
    "sb": torch.ones((2, 32), dtype=torch.float8_e4m3fn),
    "out_dtype": torch.bfloat16,
    "block_size": 16,
    "bias": None,
}

NVFP4_SUPPORT_CASES = (
    ({}, True),
    ({"aq": NVFP4_SUPPORT_INPUTS["aq"][0]}, False),
    ({"aq": torch.zeros((32, 32), dtype=torch.uint8)[:, ::2]}, False),
    ({"bq": torch.zeros((32, 32), dtype=torch.uint8)[::2]}, False),
    ({"aq": NVFP4_SUPPORT_INPUTS["aq"].to(torch.int8)}, False),
    ({"bq": NVFP4_SUPPORT_INPUTS["bq"].to(torch.int8)}, False),
    ({"out_dtype": torch.float8_e4m3fn}, False),
    ({"block_size": 32}, False),
    (
        {
            "aq": torch.zeros((32, 8), dtype=torch.uint8),
            "bq": torch.zeros((8, 32), dtype=torch.uint8),
            "sa": torch.ones((32, 1), dtype=torch.float8_e4m3fn),
            "sb": torch.ones((1, 32), dtype=torch.float8_e4m3fn),
        },
        False,
    ),
    (
        {
            "bq": torch.zeros((15, 16), dtype=torch.uint8).t(),
            "sb": torch.ones((2, 15), dtype=torch.float8_e4m3fn),
        },
        False,
    ),
    ({"sa": NVFP4_SUPPORT_INPUTS["sa"].to(torch.float16)}, False),
    ({"sb": torch.ones((1, 32), dtype=torch.float8_e4m3fn)}, False),
    ({"out_dtype": torch.float32, "bias": torch.zeros(32, dtype=torch.float32)}, False),
    ({"bias": torch.zeros(32, dtype=torch.float16)}, False),
    ({"bias": torch.zeros(31, dtype=torch.bfloat16)}, False),
    ({"bias": torch.zeros(64, dtype=torch.bfloat16)[::2]}, False),
)


@pytest.mark.parametrize("overrides,supported", MXFP8_SUPPORT_CASES)
def test_supports_mxfp8_scaled_mm(overrides, supported):
    inputs = MXFP8_SUPPORT_INPUTS | overrides
    assert cublaslt_mm.supports_mxfp8_scaled_mm(**inputs) is supported


@pytest.mark.parametrize("overrides,supported", NVFP4_SUPPORT_CASES)
def test_supports_nvfp4_scaled_mm(overrides, supported):
    inputs = NVFP4_SUPPORT_INPUTS | overrides
    assert cublaslt_mm.supports_nvfp4_scaled_mm(**inputs) is supported


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES)
@pytest.mark.parametrize("scale", MXFP8_SCALE_CASES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("with_global_scale", (False, True))
def test_mxfp8_scaled_mm_precision(case, format, scale, with_bias, with_global_scale):
    # the kernel returns bf16 and rejects every other request by contract, so
    # unlike the Triton tests there is no out_dtype axis to bound against
    aq, bq, sa, sb, out_dtype, block_size, bias, gsa, gsb = make_scaled_mm_inputs(
        case,
        format,
        scale,
        scale_dtype=torch.float8_e8m0fnu,
        with_bias=with_bias,
        with_global_scale=with_global_scale,
    )
    actual = cublaslt_mm.scaled_mm_mxfp8(
        aq, bq, sa, sb, out_dtype, block_size, bias, gsa, gsb
    )
    expected = eager_mm.scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias, gsa, gsb)

    assert actual.shape == expected.shape
    assert actual.dtype is BF16_OUT.dtype
    assert actual.stride() == (actual.shape[1], 1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=BF16_OUT.rtol, atol=BF16_OUT.atol)
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-4, rel


@cuda_only
@pytest.mark.parametrize("case", MXFP8_SCALED_MM_CASES)
@pytest.mark.parametrize("format", MXFP8_FORMAT_CASES)
@pytest.mark.parametrize("scale", MXFP8_SCALE_ERROR_CASES)
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
    message = f"MXFP8 GEMM requires block_size=32, got {block_size}"
    with pytest.raises(ValueError, match=message):
        cublaslt_mm.scaled_mm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias)


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("case", NVFP4_SCALED_MM_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("out", OUT_DTYPE_CASES)
@pytest.mark.parametrize("with_bias", BIAS_CASES)
@pytest.mark.parametrize("with_global_scale", (False, True), ids=["local", "global"])
def test_nvfp4_scaled_mm_precision(case, out, with_bias, with_global_scale):
    if case.k % 32 != 0 or case.n % 16 != 0:
        pytest.skip("cuBLASLt NVFP4 requires K divisible by 32 and N divisible by 16")
    aq, bq, sa, sb, out_dtype, block_size, bias, gsa, gsb = make_scaled_mm_inputs(
        case,
        with_bias=with_bias,
        out_dtype=out.dtype,
        with_global_scale=with_global_scale,
        packed_e2m1=True,
    )
    actual = cublaslt_mm.nvfp4_scaled_mm(
        aq, bq, sa, sb, out_dtype, block_size, bias, gsa, gsb
    )
    expected = eager_mm.scaled_mm(
        aq,
        bq,
        sa,
        sb,
        out_dtype,
        block_size,
        bias,
        gsa,
        gsb,
        unpack_e2m1=True,
    )

    assert actual.shape == expected.shape
    assert actual.dtype is out_dtype
    assert actual.stride() == (actual.shape[1], 1)
    torch.testing.assert_close(actual, expected, rtol=out.rtol, atol=out.atol)
