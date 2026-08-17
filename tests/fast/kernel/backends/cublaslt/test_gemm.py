"""Native cuBLASLt GEMM operator and adapter tests."""

import pytest
import torch

import src.kernel.backends.cublaslt._C  # noqa: F401
from src.kernel.backends.cublaslt import gemm as cublaslt_gemm
from src.kernel.ops.gemm import mxfp8_scaled_grouped_mm, mxfp8_scaled_mm
from src.quant.quantize import quantize_operand
from src.kernel.utils import to_column_major, to_swizzle_32_4_4


_MXFP8_SCALING = {
    "granularity": "blockwise",
    "block_shape": (1, 32),
    "scale_dtype": torch.float8_e8m0fnu,
}

cuda_mxfp8_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuBLASLt MXFP8 GEMM requires CUDA"
)


def _require_mxfp8_device() -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuBLASLt MXFP8 GEMM requires SM100 or newer")


def _require_grouped_mxfp8_device() -> None:
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10 and capability != (11, 0):
        pytest.skip("cuBLASLt grouped MXFP8 GEMM requires SM10x or SM110")


def _mxfp8_operands(
    shape: tuple[int, int, int], scale_case: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K, N = shape
    blocks = (K + 31) // 32
    if scale_case == "unit":
        block = torch.linspace(-256.0, 256.0, 32, device="cuda", dtype=torch.bfloat16)
        a = block.repeat(M, blocks)[:, :K].contiguous()
        b = block[:, None].repeat(blocks, N)[:K].contiguous()
    else:
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    aq, scale_a = quantize_operand(a, -1, "fp8_e4m3", _MXFP8_SCALING)
    bq, scale_b = quantize_operand(b, -2, "fp8_e4m3", _MXFP8_SCALING)
    if scale_case == "unit":
        assert torch.equal(scale_a, torch.ones_like(scale_a))
        assert torch.equal(scale_b, torch.ones_like(scale_b))
    else:
        assert scale_a.unique().numel() > 1
        assert scale_b.unique().numel() > 1
    return aq, bq, scale_a, scale_b


def _assert_matches_eager(
    aq: torch.Tensor,
    bq: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    eager = mxfp8_scaled_mm(
        aq,
        bq,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        bias=bias,
        backend="eager",
    )
    actual = mxfp8_scaled_mm(
        aq,
        bq,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        bias=bias,
        backend="cublaslt",
    )
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert actual.stride() == (actual.shape[1], 1)
    difference = actual.float() - eager.float()
    relative_error = (difference.norm() / eager.float().norm()).item()
    maximum_error = difference.abs().max().item()
    assert torch.equal(actual, eager), (relative_error, maximum_error)


@cuda_mxfp8_only
@pytest.mark.parametrize("shape", [(128, 128, 128), (129, 144, 160), (256, 384, 512)])
@pytest.mark.parametrize("scale_case", ["unit", "varying"])
def test_cublaslt_mxfp8_precision_matches_eager_bitwise(
    shape: tuple[int, int, int], scale_case: str
):
    _require_mxfp8_device()
    _assert_matches_eager(*_mxfp8_operands(shape, scale_case))


@cuda_mxfp8_only
def test_cublaslt_mxfp8_bias_matches_eager_bitwise():
    _require_mxfp8_device()
    aq, bq, scale_a, scale_b = _mxfp8_operands((129, 144, 160), "varying")
    bias = torch.linspace(-128.0, 128.0, 160, device="cuda", dtype=torch.bfloat16)
    without_bias = mxfp8_scaled_mm(
        aq,
        bq,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        backend="eager",
    )
    with_bias = mxfp8_scaled_mm(
        aq,
        bq,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        bias=bias,
        backend="eager",
    )
    assert not torch.equal(with_bias, without_bias)
    _assert_matches_eager(aq, bq, scale_a, scale_b, bias)


def test_cublaslt_adapter_has_meta_implementation():
    a = torch.empty((129, 144), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((144, 160), device="meta", dtype=torch.float8_e4m3fn)
    scale_a = torch.empty((129, 5), device="meta")
    scale_b = torch.empty((5, 160), device="meta")
    out = cublaslt_gemm.scaled_gemm_mxfp8(
        a,
        b,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
    )
    assert out.shape == (129, 160)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "meta"


def test_cublaslt_grouped_adapter_has_meta_implementation():
    aq = torch.empty((299, 144), device="meta", dtype=torch.float8_e4m3fn)
    bq = torch.empty((4, 144, 160), device="meta", dtype=torch.float8_e4m3fn)
    scale_a = torch.empty((299, 5), device="meta")
    scale_b = torch.empty((4, 5, 160), device="meta")
    offsets = torch.empty((4,), device="meta", dtype=torch.int32)

    out = cublaslt_gemm.scaled_grouped_gemm_mxfp8(
        aq,
        bq,
        scale_a,
        scale_b,
        offsets,
        torch.bfloat16,
        32,
    )

    assert out.shape == (299, 160)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "meta"


@cuda_mxfp8_only
def test_cublaslt_grouped_native_op_validates_scale_shape():
    aq = torch.empty((16, 32), device="cuda", dtype=torch.float8_e4m3fn)
    bq = torch.empty((1, 32, 16), device="cuda", dtype=torch.float8_e4m3fn)
    bq = bq.transpose(-1, -2).contiguous().transpose(-1, -2)
    scale_a = torch.empty((16, 2), device="cuda")
    scale_b = torch.empty((1, 1, 16), device="cuda")
    offsets = torch.tensor([16], device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="A scales must have shape"):
        torch.ops.aot_kernel._scaled_grouped_gemm_mxfp8_cublaslt(
            aq, bq, scale_a, scale_b, offsets
        )


@cuda_mxfp8_only
def test_cublaslt_grouped_mxfp8_matches_eager_with_empty_expert():
    _require_grouped_mxfp8_device()
    counts = [128, 0, 130, 41]
    rows, k, n = sum(counts), 144, 160
    torch.manual_seed(0)
    a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(len(counts), k, n, device="cuda", dtype=torch.bfloat16)
    aq, scale_a = quantize_operand(a, -1, "fp8_e4m3", _MXFP8_SCALING)
    bq, scale_b = quantize_operand(b, -2, "fp8_e4m3", _MXFP8_SCALING)
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0)

    eager = mxfp8_scaled_grouped_mm(
        aq,
        bq,
        scale_a,
        scale_b,
        offsets,
        torch.bfloat16,
        32,
        backend="eager",
    )
    actual = mxfp8_scaled_grouped_mm(
        aq,
        bq,
        scale_a,
        scale_b,
        offsets,
        torch.bfloat16,
        32,
        backend="cublaslt",
    )

    assert actual.shape == (rows, n)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, eager)


def test_cublaslt_meta_rejects_invalid_contraction():
    a = torch.empty((128, 128), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((64, 128), device="meta", dtype=torch.float8_e4m3fn)
    scale_a = torch.empty((512,), device="meta", dtype=torch.float8_e8m0fnu)
    scale_b = torch.empty((512,), device="meta", dtype=torch.float8_e8m0fnu)
    with pytest.raises(RuntimeError, match="contraction dimensions"):
        torch.ops.aot_kernel._scaled_gemm_mxfp8_cublaslt(a, b, scale_a, scale_b)


@pytest.mark.parametrize(
    ("invalid_input", "error"),
    [
        ("scale_dtype", "A scale must have float8_e8m0fnu dtype"),
        ("short_scale", "A scale buffer is too small"),
        ("invalid_bias", "bias must have shape"),
    ],
)
def test_cublaslt_meta_rejects_invalid_transformed_abi(invalid_input: str, error: str):
    a = torch.empty((128, 128), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((128, 128), device="meta", dtype=torch.float8_e4m3fn)
    scale_a = torch.empty((512,), device="meta", dtype=torch.float8_e8m0fnu)
    scale_b = torch.empty((512,), device="meta", dtype=torch.float8_e8m0fnu)
    bias = None
    if invalid_input == "scale_dtype":
        scale_a = scale_a.float()
    elif invalid_input == "short_scale":
        scale_a = scale_a[:-1]
    else:
        bias = torch.empty((127,), device="meta", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match=error):
        torch.ops.aot_kernel._scaled_gemm_mxfp8_cublaslt(a, b, scale_a, scale_b, bias)


@cuda_mxfp8_only
@pytest.mark.parametrize(
    ("invalid_input", "error"),
    [
        ("row_major_b", "B must be column-major contiguous"),
        ("short_scale", "A scale buffer is too small"),
        ("rank_two_scale", "A scale must be a contiguous rank-1"),
        ("invalid_bias", "bias must have shape"),
    ],
)
def test_cublaslt_native_operator_rejects_invalid_transformed_abi(
    invalid_input: str, error: str
):
    _require_mxfp8_device()
    aq, bq, scale_a, scale_b = _mxfp8_operands((128, 128, 128), "varying")
    b = to_column_major(bq)
    swizzled_a = to_swizzle_32_4_4(scale_a)
    swizzled_b = to_swizzle_32_4_4(scale_b.t())
    bias = None

    if invalid_input == "row_major_b":
        b = b.contiguous()
    elif invalid_input == "short_scale":
        swizzled_a = swizzled_a[:-1]
    elif invalid_input == "rank_two_scale":
        swizzled_a = swizzled_a.reshape(1, -1)
    else:
        bias = torch.empty(127, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match=error):
        torch.ops.aot_kernel._scaled_gemm_mxfp8_cublaslt(
            aq, b, swizzled_a, swizzled_b, bias
        )
