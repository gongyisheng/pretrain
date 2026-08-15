from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from src.kernel.registry import register_kernel
from src.kernel.spec import CheckResult
from src.kernel.utils import (
    check_compute_capability_at_least,
    check_contiguous,
    check_cuda_tensors,
    check_dtypes,
    check_matrix_layout,
    check_rank,
    check_same_device,
)

try:
    from src.kernel.backends.cublaslt import _C  # noqa: F401
except ImportError as error:
    _EXTENSION_ERROR: str | None = f"cuBLASLt extension unavailable: {error}"
else:
    _EXTENSION_ERROR = None


_E4M3 = frozenset({torch.float8_e4m3fn})
_FP32 = frozenset({torch.float32})


def _first_failure(results: tuple[CheckResult, ...]) -> CheckResult:
    for result in results:
        if not result.ok:
            return result
    return CheckResult(True)


def _swizzle_32_4_4(scale: torch.Tensor) -> torch.Tensor:
    rows, blocks = scale.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_blocks = (blocks + 3) // 4 * 4
    padded = F.pad(scale, (0, padded_blocks - blocks, 0, padded_rows - rows))
    return (
        padded.to(torch.float8_e8m0fnu)
        .reshape(padded_rows // 128, 4, 32, padded_blocks // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )


def _column_major(bq: torch.Tensor) -> torch.Tensor:
    if bq.stride(0) == 1:
        return bq
    return bq.t().contiguous().t()


def can_implement_scaled_gemm_mxfp8(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    if _EXTENSION_ERROR is not None:
        return CheckResult(False, _EXTENSION_ERROR)
    if len(args) != 8:
        return CheckResult(
            False, "cuBLASLt MXFP8 GEMM requires eight normalized arguments"
        )
    aq, bq, sa, sb, out_dtype, block_size, bias, scale_dtype = args
    tensors = (aq, bq, sa, sb) if bias is None else (aq, bq, sa, sb, bias)
    result = _first_failure(
        (
            check_same_device(tensors, "cuBLASLt MXFP8 GEMM"),
            check_cuda_tensors(tensors, "cuBLASLt MXFP8 GEMM"),
            check_dtypes((aq, bq), _E4M3, "cuBLASLt MXFP8 GEMM operands"),
            check_rank(aq, frozenset({2}), "cuBLASLt MXFP8 GEMM A"),
            check_rank(bq, frozenset({2}), "cuBLASLt MXFP8 GEMM B"),
            check_rank(sa, frozenset({2}), "cuBLASLt MXFP8 GEMM A scale"),
            check_rank(sb, frozenset({2}), "cuBLASLt MXFP8 GEMM B scale"),
            check_dtypes((sa, sb), _FP32, "cuBLASLt MXFP8 GEMM scales"),
            check_contiguous(sa, "cuBLASLt MXFP8 GEMM A scale"),
            check_contiguous(sb, "cuBLASLt MXFP8 GEMM B scale"),
            check_matrix_layout(aq, 16, "cuBLASLt MXFP8 GEMM A"),
            check_matrix_layout(bq, 16, "cuBLASLt MXFP8 GEMM B"),
        )
    )
    if not result.ok:
        return result
    if aq.numel() and aq.stride() != (aq.shape[1], 1):
        return CheckResult(False, "cuBLASLt MXFP8 GEMM requires compact row-major A")
    if bq.numel() and bq.stride() not in (
        (bq.shape[1], 1),
        (1, bq.shape[0]),
    ):
        return CheckResult(
            False,
            "cuBLASLt MXFP8 GEMM requires compact row-major or column-major B",
        )
    if aq.shape[1] != bq.shape[0]:
        return CheckResult(
            False, "cuBLASLt MXFP8 GEMM contraction dimensions must match"
        )
    if any(dimension % 16 != 0 for dimension in (aq.shape[1], bq.shape[1])):
        return CheckResult(
            False, "cuBLASLt MXFP8 GEMM dimensions must be multiples of 16"
        )
    if block_size != 32:
        return CheckResult(False, "cuBLASLt MXFP8 GEMM requires block_size 32")
    if scale_dtype != "fp8_e8m0":
        return CheckResult(False, "cuBLASLt MXFP8 GEMM requires scale_dtype fp8_e8m0")
    scale_blocks = (aq.shape[1] + 31) // 32
    if sa.shape != (aq.shape[0], scale_blocks) or sb.shape != (
        scale_blocks,
        bq.shape[1],
    ):
        return CheckResult(
            False, "cuBLASLt MXFP8 GEMM requires logical 1x32 scale shapes"
        )
    if out_dtype != torch.bfloat16:
        return CheckResult(False, "cuBLASLt MXFP8 GEMM output must be bf16")
    if bias is not None:
        if bias.dtype != torch.bfloat16:
            return CheckResult(False, "cuBLASLt MXFP8 GEMM bias dtype must be bf16")
        result = _first_failure(
            (
                check_rank(bias, frozenset({1}), "cuBLASLt MXFP8 GEMM bias"),
                check_contiguous(bias, "cuBLASLt MXFP8 GEMM bias"),
            )
        )
        if not result.ok:
            return result
        if bias.shape != (bq.shape[1],):
            return CheckResult(
                False, "cuBLASLt MXFP8 GEMM bias must be an output-width vector"
            )
    return check_compute_capability_at_least(aq.device, (10, 0), "cuBLASLt MXFP8 GEMM")


@register_kernel(
    op="gemm.scaled",
    backend="cublaslt",
    priority=150,
    can_implement=can_implement_scaled_gemm_mxfp8,
    build="aot",
    autograd=False,
)
def scaled_gemm_mxfp8(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    bias=None,
    scale_dtype=None,
):
    del out_dtype, block_size, scale_dtype
    return torch.ops.aot_kernel._scaled_gemm_mxfp8_cublaslt(
        aq,
        _column_major(bq),
        _swizzle_32_4_4(sa),
        _swizzle_32_4_4(sb.t()),
        bias,
    )
