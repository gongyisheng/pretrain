from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from src.kernel.registry import register_kernel
from src.kernel.spec import CheckResult
from src.kernel.utils import (
    BF16,
    FP32,
    FP8_E4M3,
    FP8_E8M0,
    check_condition,
    check_compute_capability_at_least,
    check_contiguous,
    check_cuda_tensors,
    check_dimension_size_match,
    check_dimension_sizes_multiple_of,
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
    result = first_failure(
        (
            check_condition(
                _EXTENSION_ERROR is None,
                str(_EXTENSION_ERROR),
            ),
            check_condition(
                len(args) == 8,
                "cuBLASLt MXFP8 GEMM requires eight normalized arguments",
            ),
        )
    )
    if not result.ok:
        return result
    aq, bq, sa, sb, out_dtype, block_size, bias, scale_dtype = args
    tensors = (aq, bq, sa, sb) if bias is None else (aq, bq, sa, sb, bias)
    result = first_failure(
        (
            check_same_device(tensors, "cuBLASLt MXFP8 GEMM"),
            check_cuda_tensors(tensors, "cuBLASLt MXFP8 GEMM"),
            check_dtypes((aq, bq), FP8_E4M3, "cuBLASLt MXFP8 GEMM operands"),
            check_ndim(aq, frozenset({2}), "cuBLASLt MXFP8 GEMM A"),
            check_ndim(bq, frozenset({2}), "cuBLASLt MXFP8 GEMM B"),
            check_ndim(sa, frozenset({2}), "cuBLASLt MXFP8 GEMM A scale"),
            check_ndim(sb, frozenset({2}), "cuBLASLt MXFP8 GEMM B scale"),
            check_dtypes((sa, sb), FP32, "cuBLASLt MXFP8 GEMM scales"),
            check_values(
                (block_size,),
                frozenset({32}),
                "cuBLASLt MXFP8 GEMM block_size",
            ),
            check_dtypes((scale_dtype,), FP8_E8M0, "cuBLASLt MXFP8 GEMM scale_dtype"),
        )
    )
    if not result.ok:
        return result
    a_allowed_strides = ((aq.shape[1], 1),)
    b_allowed_strides = ((bq.shape[1], 1), (1, bq.shape[0]))
    scale_blocks = (aq.shape[1] + 31) // 32
    bias_results = (
        ()
        if bias is None
        else (
            check_dtypes((bias,), BF16, "cuBLASLt MXFP8 GEMM bias"),
            check_ndim(bias, frozenset({1}), "cuBLASLt MXFP8 GEMM bias"),
            check_contiguous(bias, "cuBLASLt MXFP8 GEMM bias"),
            check_shape(bias, (bq.shape[1],), "cuBLASLt MXFP8 GEMM bias"),
        )
    )
    result = first_failure(
        (
            check_dimension_size_match(
                ((aq, -1), (bq, -2)),
                "cuBLASLt MXFP8 GEMM contraction dimensions",
            ),
            check_contiguous(sa, "cuBLASLt MXFP8 GEMM A scale"),
            check_contiguous(sb, "cuBLASLt MXFP8 GEMM B scale"),
            check_matrix_layout(aq, 16, "cuBLASLt MXFP8 GEMM A"),
            check_matrix_layout(bq, 16, "cuBLASLt MXFP8 GEMM B"),
            check_stride(aq, a_allowed_strides, "cuBLASLt MXFP8 GEMM A"),
            check_stride(bq, b_allowed_strides, "cuBLASLt MXFP8 GEMM B"),
            check_dimension_sizes_multiple_of(
                ((aq, -1), (bq, -1)), 16, "cuBLASLt MXFP8 GEMM dimensions"
            ),
            check_shape(sa, (aq.shape[0], scale_blocks), "cuBLASLt MXFP8 GEMM A scale"),
            check_shape(sb, (scale_blocks, bq.shape[1]), "cuBLASLt MXFP8 GEMM B scale"),
            check_dtypes((out_dtype,), BF16, "cuBLASLt MXFP8 GEMM output dtype"),
            *bias_results,
        )
    )
    if not result.ok:
        return result
    return check_compute_capability_at_least(aq.device, (10, 0), "cuBLASLt MXFP8 GEMM")


@register_kernel(
    op="gemm.scaled",
    backend="cublaslt",
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
