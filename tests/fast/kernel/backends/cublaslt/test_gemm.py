"""cuBLASLt GEMM backend tests."""

import pytest
import torch

import src.kernel.backends.cublaslt._C  # noqa: F401
from src.kernel.backends.cublaslt import gemm as cublaslt_gemm
from src.kernel.ops.gemm import mxfp8_scaled_mm
from src.kernel.selector import select_kernel
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
    out = cublaslt_gemm.scaled_mm_mxfp8(
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


def test_cublaslt_rejects_non_bf16_out_dtype():
    a = torch.empty((129, 144), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((144, 160), device="meta", dtype=torch.float8_e4m3fn)
    scale_a = torch.empty((129, 5), device="meta")
    scale_b = torch.empty((5, 160), device="meta")

    with pytest.raises(ValueError, match="requires out_dtype=bfloat16"):
        cublaslt_gemm.scaled_mm_mxfp8(a, b, scale_a, scale_b, torch.float32, 32)


def test_cublaslt_meta_rejects_invalid_contraction():
    a = torch.empty((128, 128), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((64, 128), device="meta", dtype=torch.float8_e4m3fn)
    scale_a = torch.empty((512,), device="meta", dtype=torch.float8_e8m0fnu)
    scale_b = torch.empty((512,), device="meta", dtype=torch.float8_e8m0fnu)
    with pytest.raises(RuntimeError, match="contraction dimensions"):
        torch.ops.aot_kernel._scaled_mm_mxfp8_cublaslt(a, b, scale_a, scale_b)


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
        torch.ops.aot_kernel._scaled_mm_mxfp8_cublaslt(a, b, scale_a, scale_b, bias)


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
        torch.ops.aot_kernel._scaled_mm_mxfp8_cublaslt(
            aq, b, swizzled_a, swizzled_b, bias
        )


@cuda_mxfp8_only
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
        cublaslt_gemm.supports_mxfp8_scaled_mm(aq, bq, out_dtype, block_size)
        is expected
    )


def _fp8_values(count: int, dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    values = (torch.arange(count, device="cuda", dtype=torch.float32) % 17 - 8) / 16
    return values.to(dtype)


def _view_operands(case: str) -> tuple[object, ...]:
    m, k, n = 16, 32, 16
    a = _fp8_values(m * k).view(m, k)
    b = _fp8_values(k * n).view(k, n)
    if case == "padded_a":
        a = _fp8_values(m * (k + 16)).view(m, k + 16)[:, :k]
        assert a.stride() == (k + 16, 1)
    elif case == "padded_b":
        b = _fp8_values(n * (k + 16)).view(n, k + 16)[:, :k].t()
        assert b.stride() == (1, k + 16)
    elif case == "offset_a":
        a = _fp8_values(m * k + 1)[1:].view(m, k)
        assert a.data_ptr() % 16 != 0
    else:
        b = _fp8_values(n * k + 1)[1:].view(n, k).t()
        assert b.data_ptr() % 16 != 0
    scale_a = torch.ones((m, 1), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((1, n), device="cuda", dtype=torch.float32)
    return a, b, scale_a, scale_b, torch.bfloat16, 32, None


@cuda_mxfp8_only
@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("padded_a", "MXFP8 GEMM A has stride"),
        ("padded_b", "MXFP8 GEMM B has stride"),
    ],
)
def test_forced_backend_rejects_unsafe_operand_view(case: str, reason: str):
    _require_mxfp8_device()
    args = _view_operands(case)

    with pytest.raises(ValueError, match=reason):
        mxfp8_scaled_mm(*args, backend="cublaslt")


@cuda_mxfp8_only
def test_forced_backend_rejects_bad_scale_config():
    _require_mxfp8_device()
    m, k, n = 16, 32, 16
    args = (
        _fp8_values(m * k).view(m, k),
        _fp8_values(k * n).view(k, n),
        torch.ones((m, 1), device="cuda", dtype=torch.float32),
        torch.ones((1, n), device="cuda", dtype=torch.float32),
        torch.bfloat16,
        64,
        None,
    )

    with pytest.raises(ValueError, match="requires block_size=32"):
        mxfp8_scaled_mm(*args, backend="cublaslt")


@cuda_mxfp8_only
@pytest.mark.parametrize("case", ["offset_a", "offset_b"])
def test_misaligned_operand_executes_with_cublaslt(case: str):
    _require_mxfp8_device()
    args = _view_operands(case)

    selected = select_kernel("gemm.mxfp8_scaled_mm", backend="cublaslt")
    expected = mxfp8_scaled_mm(*args, backend="eager")
    actual = mxfp8_scaled_mm(*args, backend="cublaslt")

    assert selected.backend == "cublaslt"
    assert torch.equal(actual, expected)


def _zero_operands(shape: tuple[int, int, int], with_bias: bool) -> tuple[object, ...]:
    m, k, n = shape
    blocks = (k + 31) // 32
    a = torch.empty((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty((k, n), device="cuda", dtype=torch.float8_e4m3fn)
    scale_a = torch.ones((m, blocks), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((blocks, n), device="cuda", dtype=torch.float32)
    bias = (
        torch.linspace(-1, 1, n, device="cuda", dtype=torch.bfloat16)
        if with_bias
        else None
    )
    return a, b, scale_a, scale_b, torch.bfloat16, 32, bias


@cuda_mxfp8_only
@pytest.mark.parametrize("backend", [None, "cublaslt"])
@pytest.mark.parametrize(
    ("shape", "with_bias"),
    [
        ((0, 16, 16), False),
        ((16, 16, 0), False),
        ((16, 0, 16), False),
        ((16, 0, 16), True),
    ],
)
def test_zero_size_gemm_matches_eager(
    backend: str | None, shape: tuple[int, int, int], with_bias: bool
):
    _require_mxfp8_device()
    args = _zero_operands(shape, with_bias)

    expected = mxfp8_scaled_mm(*args, backend="eager")
    actual = mxfp8_scaled_mm(*args, backend=backend)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
