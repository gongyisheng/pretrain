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
    check_dimension_sizes_match,
    check_dtypes,
    check_matrix_layout,
    check_ndim,
    check_same_device,
    check_shape,
    check_stride,
    check_values,
    first_failure,
)

try:
    from src.kernel.backends.cublaslt import _C  # noqa: F401
except ImportError as error:
    _EXTENSION_ERROR: str | None = f"cuBLASLt extension unavailable: {error}"
else:
    _EXTENSION_ERROR = None


_E4M3 = frozenset({torch.float8_e4m3fn})
_FP32 = frozenset({torch.float32})


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
    a_allowed_strides = ((aq.shape[1], 1),) if aq.ndim == 2 else ()
    b_allowed_strides = ((bq.shape[1], 1), (1, bq.shape[0])) if bq.ndim == 2 else ()
    result = first_failure(
        (
            check_same_device(tensors, "cuBLASLt MXFP8 GEMM"),
            check_cuda_tensors(tensors, "cuBLASLt MXFP8 GEMM"),
            check_dtypes((aq, bq), _E4M3, "cuBLASLt MXFP8 GEMM operands"),
            check_ndim(aq, frozenset({2}), "cuBLASLt MXFP8 GEMM A"),
            check_ndim(bq, frozenset({2}), "cuBLASLt MXFP8 GEMM B"),
            check_dimension_sizes_match(
                ((aq, -1), (bq, -2)),
                "cuBLASLt MXFP8 GEMM contraction dimensions",
            ),
            check_ndim(sa, frozenset({2}), "cuBLASLt MXFP8 GEMM A scale"),
            check_ndim(sb, frozenset({2}), "cuBLASLt MXFP8 GEMM B scale"),
            check_dtypes((sa, sb), _FP32, "cuBLASLt MXFP8 GEMM scales"),
            check_contiguous(sa, "cuBLASLt MXFP8 GEMM A scale"),
            check_contiguous(sb, "cuBLASLt MXFP8 GEMM B scale"),
            check_matrix_layout(aq, 16, "cuBLASLt MXFP8 GEMM A"),
            check_matrix_layout(bq, 16, "cuBLASLt MXFP8 GEMM B"),
            check_stride(aq, a_allowed_strides, "cuBLASLt MXFP8 GEMM A"),
            check_stride(bq, b_allowed_strides, "cuBLASLt MXFP8 GEMM B"),
            check_compute_capability_at_least(
                aq.device, (10, 0), "cuBLASLt MXFP8 GEMM"
            ),
        )
    )
    if not result.ok:
        return result
    if any(dimension % 16 != 0 for dimension in (aq.shape[1], bq.shape[1])):
        return CheckResult(
            False, "cuBLASLt MXFP8 GEMM dimensions must be multiples of 16"
        )
    result = check_values(
        (block_size,), frozenset({32}), "cuBLASLt MXFP8 GEMM block_size"
    )
    if not result.ok:
        return result
    result = check_values(
        (scale_dtype,),
        frozenset({"fp8_e8m0"}),
        "cuBLASLt MXFP8 GEMM scale_dtype",
    )
    if not result.ok:
        return result
    scale_blocks = (aq.shape[1] + 31) // 32
    result = first_failure(
        (
            check_shape(sa, (aq.shape[0], scale_blocks), "cuBLASLt MXFP8 GEMM A scale"),
            check_shape(sb, (scale_blocks, bq.shape[1]), "cuBLASLt MXFP8 GEMM B scale"),
        )
    )
    if not result.ok:
        return result
    result = check_dtypes(
        (out_dtype,),
        frozenset({torch.bfloat16}),
        "cuBLASLt MXFP8 GEMM output dtype",
    )
    if not result.ok:
        return result
    if bias is not None:
        result = first_failure(
            (
                check_dtypes(
                    (bias,), frozenset({torch.bfloat16}), "cuBLASLt MXFP8 GEMM bias"
                ),
                check_ndim(bias, frozenset({1}), "cuBLASLt MXFP8 GEMM bias"),
                check_contiguous(bias, "cuBLASLt MXFP8 GEMM bias"),
                check_shape(bias, (bq.shape[1],), "cuBLASLt MXFP8 GEMM bias"),
            )
        )
        if not result.ok:
            return result
    return CheckResult(True)


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
