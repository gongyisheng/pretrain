"""Torch GEMM parity against the eager correctness backend."""

import pytest
import torch
import torch.nn.functional as F

from src.kernel.backends.eager import gemm as eager_gemm
from src.kernel.backends.torch import gemm as torch_gemm
from src.quant.quantize import quantize_operand
from tests.fast.kernel.backends.helper import (
    BIAS_CASES,
    DENSE_WORKLOADS,
    FORMAT_PAIRS,
    GROUPED_LAYOUTS,
    GROUPED_PROJECTIONS,
    SCALE_DTYPE_VALUES,
    SCALE_DTYPES,
    SCALING_CASES,
    FormatPair,
    ScalingCase,
    make_grouped_inputs,
    make_scaled_gemm_inputs,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Torch GEMM kernels are CUDA only"
)


@pytest.mark.parametrize("projection", GROUPED_PROJECTIONS, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.skipif(
    torch.cuda.is_available()
    and (
        not callable(getattr(F, "grouped_mm", None))
        or torch.cuda.get_device_capability() < (8, 0)
    ),
    reason="Torch grouped GEMM requires torch.nn.functional.grouped_mm and SM80+",
)
def test_grouped_gemm_precision(projection, layout, with_bias):
    if layout == "ragged_n" and with_bias:
        pytest.skip("ragged-N has no defined per-expert output-channel bias")
    a, b, offs, bias = make_grouped_inputs(projection, layout, with_bias)

    actual = torch_gemm.grouped_gemm(a, b, offs, bias)
    expected = eager_gemm.grouped_gemm(a, b, offs, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def torch_scaled_skip_reason(
    format_pair: FormatPair, scaling_case: ScalingCase
) -> str | None:
    if format_pair.name == "e4m3xe5m2":
        if scaling_case.block_size == 0:
            return "public block_size=0 scale layout cannot distinguish tensorwise from rowwise for this PyTorch recipe"
        return "F.scaled_mm does not support E5M2 as the B operand"
    if format_pair.name == "e5m2xe5m2":
        return "F.scaled_mm does not support E5M2 x E5M2"
    return None


@pytest.mark.parametrize("workload", DENSE_WORKLOADS, ids=lambda case: case.name)
@pytest.mark.parametrize("format_pair", FORMAT_PAIRS, ids=lambda case: case.name)
@pytest.mark.parametrize("scaling_case", SCALING_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("scale_dtype", SCALE_DTYPES)
@pytest.mark.parametrize("with_bias", BIAS_CASES, ids=["no-bias", "bias"])
def test_scaled_gemm_precision(
    workload, format_pair, scaling_case, scale_dtype, with_bias
):
    reason = torch_scaled_skip_reason(format_pair, scaling_case)
    if reason is not None:
        pytest.skip(reason)
    capability = torch.cuda.get_device_capability()
    if format_pair.a_format == "int8" and capability < (8, 0):
        pytest.skip("Torch INT8 scaled GEMM requires SM80+")
    if format_pair.a_format.startswith("fp8") and capability < (8, 9):
        pytest.skip("Torch FP8 scaled GEMM requires SM89+")

    aq, bq, sa, sb, out_dtype, block_size, scale_dtype, bias = make_scaled_gemm_inputs(
        workload, format_pair, scaling_case, scale_dtype, with_bias
    )
    actual = torch_gemm.scaled_gemm(
        aq, bq, sa, sb, out_dtype, block_size, scale_dtype, bias
    )
    expected = eager_gemm.scaled_gemm(
        aq, bq, sa, sb, out_dtype, block_size, scale_dtype, bias
    )

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=format_pair.rtol, atol=format_pair.atol
    )


def _make_scaled_route_inputs(
    a_format: str,
    b_format: str,
    block_size: int,
    scale_dtype: str,
    with_bias: bool,
    k: int = 64,
):
    torch.manual_seed(0)
    a = torch.randn(64, k, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(k, 48, device="cuda", dtype=torch.bfloat16) * 0.1
    scaling = {
        "granularity": "blockwise" if block_size else "rowwise",
        "block_shape": (1, block_size) if block_size else (1, 0),
        "scale_dtype": SCALE_DTYPE_VALUES[scale_dtype],
    }
    aq, sa = quantize_operand(a, -1, a_format, scaling)
    bq, sb = quantize_operand(b, -2, b_format, scaling)
    bias = (
        torch.randn(48, device="cuda", dtype=torch.bfloat16) * 0.1
        if with_bias
        else None
    )
    return (
        aq,
        bq,
        sa,
        sb,
        torch.bfloat16,
        block_size,
        SCALE_DTYPE_VALUES[scale_dtype],
        bias,
    )


def _assert_scaled_route(
    a_format: str,
    b_format: str,
    block_size: int,
    scale_dtype: str,
    with_bias: bool,
    k: int = 64,
):
    args = _make_scaled_route_inputs(
        a_format, b_format, block_size, scale_dtype, with_bias, k
    )

    actual = torch_gemm.scaled_gemm(*args)
    expected = eager_gemm.scaled_gemm(*args)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("block_size", [0, 16], ids=["rowwise", "blockwise-16"])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_scaled_route_int8(block_size, with_bias):
    _assert_scaled_route("int8", "int8", block_size, "fp32", with_bias)


@pytest.mark.parametrize(
    ("a_format", "block_size"),
    [
        pytest.param("fp8_e4m3", 0, id="e4m3xe4m3-rowwise"),
        pytest.param("fp8_e4m3", 16, id="e4m3xe4m3-blockwise-16"),
        pytest.param("fp8_e5m2", 0, id="e5m2xe4m3-rowwise"),
        pytest.param("fp8_e5m2", 32, id="e5m2xe4m3-blockwise-32"),
    ],
)
def test_scaled_route_composed_fp8(a_format, block_size):
    _assert_scaled_route(a_format, "fp8_e4m3", block_size, "fp32", True)


@pytest.mark.parametrize(
    ("block_size", "k"),
    [
        pytest.param(32, 64, id="blockwise-32"),
        pytest.param(64, 128, id="blockwise-64"),
        pytest.param(128, 256, id="blockwise-128"),
        pytest.param(128, 192, id="blockwise-128-tail-k192"),
    ],
)
def test_scaled_route_native_mxfp8(block_size, k):
    capability = torch.cuda.get_device_capability()
    if capability[0] not in (10, 12):
        pytest.skip("native Torch MXFP8 requires SM100 or SM120")
    if (
        getattr(getattr(F, "ScalingType", None), "BlockWise1x32", None) is None
        or getattr(getattr(F, "SwizzleType", None), "SWIZZLE_32_4_4", None) is None
    ):
        pytest.skip("native Torch MXFP8 scaling enums are unavailable")
    _assert_scaled_route("fp8_e4m3", "fp8_e4m3", block_size, "fp8_e8m0", True, k)
