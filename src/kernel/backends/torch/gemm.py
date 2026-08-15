from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from src.kernel.registry import register_kernel
from src.kernel.spec import CheckResult
from src.kernel.utils import (
    check_callable,
    check_compute_capability_at_least,
    check_compute_capability_in,
    check_contiguous,
    check_cuda_tensors,
    check_dtypes,
    check_matrix_layout,
    check_rank,
    check_same_device,
)


_BF16 = frozenset({torch.bfloat16})
_FP8_E4M3 = frozenset({torch.float8_e4m3fn})
_FLOAT_SCALES = frozenset({torch.float32})
_OUTPUT_DTYPES = frozenset({torch.bfloat16, torch.float16})


def _first_failure(results: tuple[CheckResult, ...]) -> CheckResult:
    for result in results:
        if not result.ok:
            return result
    return CheckResult(True)


def _check_functional(attribute: str) -> CheckResult:
    return check_callable(F, "torch.nn.functional", attribute)


def _check_rowwise_scaling_functional(attribute: str) -> CheckResult:
    result = _check_functional(attribute)
    if not result.ok:
        return result
    result = check_callable(F, "torch.nn.functional", "ScalingType")
    if not result.ok:
        return result
    if getattr(F.ScalingType, "RowWise", None) is None:
        return CheckResult(
            False,
            "torch.nn.functional.ScalingType.RowWise is unavailable",
        )
    return CheckResult(True)


def _is_row_major(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous() and tensor.stride(-1) == 1


def can_implement_grouped_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    a, b, offs, bias = args
    tensors = (a, b, offs) if bias is None else (a, b, offs, bias)
    result = _first_failure(
        (
            _check_functional("grouped_mm"),
            check_same_device(tensors, "torch grouped GEMM"),
            check_cuda_tensors(tensors, "torch grouped GEMM"),
            check_dtypes((a, b), _BF16, "torch grouped GEMM operands"),
            check_rank(a, frozenset({2, 3}), "torch grouped GEMM A"),
            check_rank(b, frozenset({2, 3}), "torch grouped GEMM B"),
            check_rank(offs, frozenset({1}), "torch grouped GEMM offsets"),
            check_contiguous(offs, "torch grouped GEMM offsets"),
            check_matrix_layout(a, 16, "torch grouped GEMM A"),
            check_matrix_layout(b, 16, "torch grouped GEMM B"),
        )
    )
    if not result.ok:
        return result
    result = check_compute_capability_at_least(a.device, (8, 0), "torch grouped GEMM")
    if not result.ok:
        return result
    if offs.dtype != torch.int32:
        return CheckResult(
            False, "torch grouped GEMM requires contiguous 1D int32 offsets"
        )
    if offs.numel() == 0:
        return CheckResult(False, "torch grouped GEMM requires at least one offset")
    if a.ndim == 3 and b.ndim == 3:
        return CheckResult(False, "3D x 3D has no ragged dimension")
    if a.ndim == 2 and b.ndim == 3:
        if a.shape[1] != b.shape[1]:
            return CheckResult(False, "grouped GEMM contraction dimensions must match")
        if b.shape[0] != offs.numel():
            return CheckResult(False, "group count must match the offset count")
    elif a.ndim == 2 and b.ndim == 2:
        if a.shape[1] != b.shape[0]:
            return CheckResult(False, "grouped GEMM contraction dimensions must match")
    else:
        if a.shape[2] != b.shape[0]:
            return CheckResult(False, "grouped GEMM contraction dimensions must match")
        if a.shape[0] != offs.numel():
            return CheckResult(False, "group count must match the offset count")
    if bias is not None:
        if a.ndim == 3:
            return CheckResult(False, "bias is not supported for the ragged-N layout")
        result = _first_failure(
            (
                check_rank(bias, frozenset({2}), "torch grouped GEMM bias"),
                check_contiguous(bias, "torch grouped GEMM bias"),
            )
        )
        if not result.ok:
            return result
        if bias.dtype != a.dtype:
            return CheckResult(False, "bias dtype must match operand dtype")
        if bias.shape != (offs.numel(), b.shape[-1]):
            return CheckResult(
                False, "bias must have shape (group count, output width)"
            )
    return CheckResult(True)


@register_kernel(
    op="gemm.grouped",
    backend="torch",
    priority=50,
    can_implement=can_implement_grouped_gemm,
    build="eager",
    autograd=False,
)
def grouped_gemm(a, b, offs, bias=None):
    out = F.grouped_mm(a, b, offs=offs, bias=None)
    if bias is None:
        return out
    if out.ndim == 3:
        return out + bias[:, None, :]
    rows = torch.arange(out.shape[0], device=offs.device)
    groups = torch.searchsorted(offs, rows, right=True)
    return out + bias[groups]


def _scaled_common_checks(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    bias,
    scale_dtype,
    feature: str,
) -> CheckResult:
    tensors = (aq, bq, sa, sb) if bias is None else (aq, bq, sa, sb, bias)
    result = _first_failure(
        (
            check_same_device(tensors, feature),
            check_cuda_tensors(tensors, feature),
            check_dtypes((aq, bq), _FP8_E4M3, f"{feature} operands"),
            check_rank(aq, frozenset({2}), f"{feature} A"),
            check_rank(bq, frozenset({2}), f"{feature} B"),
            check_rank(sa, frozenset({2}), f"{feature} A scale"),
            check_rank(sb, frozenset({2}), f"{feature} B scale"),
            check_dtypes((sa, sb), _FLOAT_SCALES, f"{feature} scales"),
            check_contiguous(sa, f"{feature} A scale"),
            check_contiguous(sb, f"{feature} B scale"),
        )
    )
    if not result.ok:
        return result
    if not _is_row_major(aq):
        return CheckResult(False, f"{feature} requires row-major A")
    if aq.shape[1] != bq.shape[0]:
        return CheckResult(False, f"{feature} contraction dimensions must match")
    if any(dimension % 16 != 0 for dimension in (aq.shape[1], bq.shape[1])):
        return CheckResult(False, f"{feature} dimensions must be multiples of 16")
    if block_size != 0:
        return CheckResult(False, f"{feature} supports rowwise scales only")
    if scale_dtype not in (None, "fp32"):
        return CheckResult(False, f"{feature} requires fp32 scales")
    if sa.shape != (aq.shape[0], 1) or sb.shape != (1, bq.shape[1]):
        return CheckResult(False, f"{feature} requires rowwise scale shapes")
    if out_dtype not in _OUTPUT_DTYPES:
        return CheckResult(False, f"{feature} output must be bf16 or fp16")
    if bias is not None and bias.dtype != out_dtype:
        return CheckResult(False, "bias dtype must match output dtype")
    if bias is not None:
        result = _first_failure(
            (
                check_rank(bias, frozenset({1}), f"{feature} bias"),
                check_contiguous(bias, f"{feature} bias"),
            )
        )
        if not result.ok:
            return result
        if bias.shape != (bq.shape[1],):
            return CheckResult(False, "bias must be a contiguous output-width vector")
    return CheckResult(True)


def can_implement_scaled_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    aq, bq, sa, sb, out_dtype, block_size, bias, scale_dtype = args
    result = _check_rowwise_scaling_functional("scaled_mm")
    if not result.ok:
        return result
    result = _scaled_common_checks(
        aq,
        bq,
        sa,
        sb,
        out_dtype,
        block_size,
        bias,
        scale_dtype,
        "torch scaled GEMM",
    )
    if not result.ok:
        return result
    return check_compute_capability_at_least(aq.device, (8, 9), "torch scaled GEMM")


def _column_major(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.stride(0) == 1:
        return tensor
    return tensor.t().contiguous().t()


@register_kernel(
    op="gemm.scaled",
    backend="torch",
    priority=50,
    can_implement=can_implement_scaled_gemm,
    build="eager",
    autograd=False,
)
def scaled_gemm(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    bias=None,
    scale_dtype=None,
):
    del block_size, scale_dtype
    return F.scaled_mm(
        aq,
        _column_major(bq),
        sa,
        F.ScalingType.RowWise,
        sb,
        F.ScalingType.RowWise,
        bias=bias,
        output_dtype=out_dtype,
    )


def can_implement_scaled_grouped_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    aq, bq, sa, sb, offs, out_dtype, block_size, bias, scale_dtype = args
    result = _check_rowwise_scaling_functional("scaled_grouped_mm")
    if not result.ok:
        return result
    tensors = (aq, bq, sa, sb, offs) if bias is None else (aq, bq, sa, sb, offs, bias)
    result = _first_failure(
        (
            check_same_device(tensors, "torch scaled grouped GEMM"),
            check_cuda_tensors(tensors, "torch scaled grouped GEMM"),
            check_dtypes((aq, bq), _FP8_E4M3, "torch scaled grouped GEMM operands"),
            check_rank(aq, frozenset({2}), "torch scaled grouped GEMM A"),
            check_rank(bq, frozenset({3}), "torch scaled grouped GEMM B"),
            check_rank(sa, frozenset({2}), "torch scaled grouped GEMM A scale"),
            check_rank(sb, frozenset({3}), "torch scaled grouped GEMM B scale"),
            check_rank(offs, frozenset({1}), "torch scaled grouped GEMM offsets"),
            check_contiguous(offs, "torch scaled grouped GEMM offsets"),
            check_dtypes((sa, sb), _FLOAT_SCALES, "torch scaled grouped GEMM scales"),
            check_contiguous(sa, "torch scaled grouped GEMM A scale"),
            check_contiguous(sb, "torch scaled grouped GEMM B scale"),
        )
    )
    if not result.ok:
        return result
    if offs.dtype != torch.int32:
        return CheckResult(
            False, "offsets must be contiguous 1D int32 on the operand device"
        )
    if offs.numel() == 0:
        return CheckResult(
            False, "torch scaled grouped GEMM requires at least one offset"
        )
    if not _is_row_major(aq):
        return CheckResult(False, "torch scaled grouped GEMM requires row-major A")
    if aq.shape[1] != bq.shape[1] or offs.numel() != bq.shape[0]:
        return CheckResult(False, "grouped operand dimensions must match offsets")
    if any(dimension % 16 != 0 for dimension in (aq.shape[1], bq.shape[2])):
        return CheckResult(
            False, "torch scaled grouped GEMM dimensions must be multiples of 16"
        )
    if block_size != 0:
        return CheckResult(
            False, "torch scaled grouped GEMM supports rowwise scales only"
        )
    if scale_dtype not in (None, "fp32"):
        return CheckResult(False, "torch scaled grouped GEMM requires fp32 scales")
    if sa.shape != (aq.shape[0], 1) or sb.shape != (bq.shape[0], 1, bq.shape[2]):
        return CheckResult(False, "torch scaled grouped GEMM requires rowwise scales")
    if out_dtype != torch.bfloat16:
        return CheckResult(False, "torch scaled grouped output must be bf16")
    if bias is not None:
        return CheckResult(False, "torch scaled grouped GEMM does not support bias")
    return check_compute_capability_in(
        aq.device,
        frozenset({9, 10}),
        "torch scaled grouped GEMM",
    )


def _group_column_major(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.stride(-2) == 1:
        return tensor
    return tensor.transpose(-1, -2).contiguous().transpose(-1, -2)


@register_kernel(
    op="gemm.scaled_grouped",
    backend="torch",
    priority=50,
    can_implement=can_implement_scaled_grouped_gemm,
    build="eager",
    autograd=False,
)
def scaled_grouped_gemm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    bias=None,
    scale_dtype=None,
):
    del block_size, scale_dtype
    return F.scaled_grouped_mm(
        aq,
        _group_column_major(bq),
        sa.squeeze(1),
        F.ScalingType.RowWise,
        sb.squeeze(1),
        F.ScalingType.RowWise,
        bias=bias,
        offs=offs,
        output_dtype=out_dtype,
    )
