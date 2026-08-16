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
    check_dimension_sizes_match,
    check_dimension_sizes_multiple_of,
    check_dtypes,
    check_matrix_layout,
    check_ndim,
    check_nonempty,
    check_same_device,
    check_shape,
    check_values,
    first_failure,
)


_BF16 = frozenset({torch.bfloat16})
_SCALED_GROUPED_DTYPES = frozenset({torch.float8_e4m3fn})
_SCALED_GEMM_DTYPE_PAIRS = frozenset(
    {
        (torch.int8, torch.int8),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e4m3fn),
    }
)
_FLOAT_SCALES = frozenset({torch.float32})
_OUTPUT_DTYPES = frozenset({torch.bfloat16, torch.float16})


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
    result = first_failure(
        (
            _check_functional("grouped_mm"),
            check_same_device(tensors, "torch grouped GEMM"),
            check_cuda_tensors(tensors, "torch grouped GEMM"),
            check_dtypes((a, b), _BF16, "torch grouped GEMM operands"),
            check_ndim(a, frozenset({2, 3}), "torch grouped GEMM A"),
            check_ndim(b, frozenset({2, 3}), "torch grouped GEMM B"),
            check_ndim(offs, frozenset({1}), "torch grouped GEMM offsets"),
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
    result = first_failure(
        (
            check_dtypes(
                (offs,), frozenset({torch.int32}), "torch grouped GEMM offsets"
            ),
            check_nonempty(offs, "torch grouped GEMM offsets"),
        )
    )
    if not result.ok:
        return result
    if a.ndim == 3 and b.ndim == 3:
        return CheckResult(False, "3D x 3D has no ragged dimension")
    result = check_dimension_sizes_match(
        ((a, -1), (b, -2)), "torch grouped GEMM contraction dimensions"
    )
    if not result.ok:
        return result
    if a.ndim == 2 and b.ndim == 3:
        result = check_dimension_sizes_match(
            ((b, 0), (offs, 0)), "torch grouped GEMM B group count"
        )
    elif a.ndim == 3:
        result = check_dimension_sizes_match(
            ((a, 0), (offs, 0)), "torch grouped GEMM A group count"
        )
    if not result.ok:
        return result
    if bias is not None:
        if a.ndim == 3:
            return CheckResult(False, "bias is not supported for the ragged-N layout")
        result = first_failure(
            (
                check_ndim(bias, frozenset({2}), "torch grouped GEMM bias"),
                check_contiguous(bias, "torch grouped GEMM bias"),
            )
        )
        if not result.ok:
            return result
        result = first_failure(
            (
                check_dtypes((bias,), frozenset({a.dtype}), "torch grouped GEMM bias"),
                check_shape(
                    bias,
                    (offs.numel(), b.shape[-1]),
                    "torch grouped GEMM bias",
                ),
            )
        )
        if not result.ok:
            return result
    return CheckResult(True)


@register_kernel(
    op="gemm.grouped",
    backend="torch",
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
    result = first_failure(
        (
            check_same_device(tensors, feature),
            check_cuda_tensors(tensors, feature),
            check_ndim(aq, frozenset({2}), f"{feature} A"),
            check_ndim(bq, frozenset({2}), f"{feature} B"),
            check_ndim(sa, frozenset({2}), f"{feature} A scale"),
            check_ndim(sb, frozenset({2}), f"{feature} B scale"),
            check_dtypes((sa, sb), _FLOAT_SCALES, f"{feature} scales"),
            check_contiguous(sa, f"{feature} A scale"),
            check_contiguous(sb, f"{feature} B scale"),
        )
    )
    if not result.ok:
        return result
    dtype_pair = (aq.dtype, bq.dtype)
    if dtype_pair not in _SCALED_GEMM_DTYPE_PAIRS:
        return CheckResult(
            False,
            f"{feature} does not support operand pair {aq.dtype} x {bq.dtype}",
        )
    if not _is_row_major(aq):
        return CheckResult(False, f"{feature} requires row-major A")
    result = check_dimension_sizes_match(
        ((aq, -1), (bq, -2)), f"{feature} contraction dimensions"
    )
    if not result.ok:
        return result
    if any(dimension == 0 for dimension in (aq.shape[0], aq.shape[1], bq.shape[1])):
        return CheckResult(False, f"{feature} dimensions must be positive")
    result = check_dimension_sizes_multiple_of(
        ((aq, -1), (bq, -1)), 16, f"{feature} dimensions"
    )
    if not result.ok:
        return result
    if type(block_size) is not int or block_size < 0:
        return CheckResult(False, "block_size must be a non-negative integer")
    result = check_values(
        (scale_dtype,),
        frozenset({None, "fp32", "fp8_e8m0"}),
        f"{feature} scale_dtype",
    )
    if not result.ok:
        return result
    scale_blocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
    result = first_failure(
        (
            check_shape(sa, (aq.shape[0], scale_blocks), f"{feature} A scale"),
            check_shape(sb, (scale_blocks, bq.shape[1]), f"{feature} B scale"),
        )
    )
    if not result.ok:
        return result
    if scale_blocks > 1 and (block_size < 16 or block_size & (block_size - 1)):
        return CheckResult(False, "block_size must be a power of two >= 16")
    result = check_dtypes((out_dtype,), _OUTPUT_DTYPES, f"{feature} output dtype")
    if not result.ok:
        return result
    if bias is not None:
        result = first_failure(
            (
                check_dtypes((bias,), frozenset({out_dtype}), f"{feature} bias"),
                check_ndim(bias, frozenset({1}), f"{feature} bias"),
                check_contiguous(bias, f"{feature} bias"),
                check_shape(bias, (bq.shape[1],), f"{feature} bias"),
            )
        )
        if not result.ok:
            return result
    return CheckResult(True)


def can_implement_scaled_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    aq, bq, sa, sb, out_dtype, block_size, bias, scale_dtype = args
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
    if aq.dtype == torch.int8 and bq.dtype == torch.int8:
        result = check_dimension_sizes_multiple_of(
            ((aq, -2),), 32, "torch scaled GEMM INT8 output rows"
        )
        if not result.ok:
            return result
        result = check_callable(torch, "torch", "_int_mm")
        minimum_capability = (8, 0)
    else:
        result = _check_rowwise_scaling_functional("scaled_mm")
        minimum_capability = (8, 9)
    if not result.ok:
        return result
    return check_compute_capability_at_least(
        aq.device, minimum_capability, "torch scaled GEMM"
    )


def _column_major(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.stride(0) == 1:
        return tensor
    return tensor.t().contiguous().t()


def _scaled_int8_mm(aq, bq, sa, sb, out_dtype, block_size, bias):
    width = block_size or aq.shape[1]
    out = torch.zeros((aq.shape[0], bq.shape[1]), device=aq.device, dtype=torch.float32)
    for index, start in enumerate(range(0, aq.shape[1], width)):
        stop = min(start + width, aq.shape[1])
        partial = torch._int_mm(
            aq[:, start:stop].contiguous(), bq[start:stop].contiguous()
        )
        out += partial.float() * sa[:, index : index + 1] * sb[index : index + 1]
    if bias is not None:
        out += bias.float()
    return out.to(out_dtype)


def _scaled_fp8_mm(aq, bq, sa, sb, out_dtype, block_size, bias):
    width = block_size or aq.shape[1]
    out = torch.zeros((aq.shape[0], bq.shape[1]), device=aq.device, dtype=torch.float32)
    for index, start in enumerate(range(0, aq.shape[1], width)):
        stop = min(start + width, aq.shape[1])
        partial = F.scaled_mm(
            aq[:, start:stop].contiguous(),
            _column_major(bq[start:stop]),
            sa[:, index : index + 1].contiguous(),
            F.ScalingType.RowWise,
            sb[index : index + 1],
            F.ScalingType.RowWise,
            bias=None,
            output_dtype=torch.bfloat16,
        )
        out += partial.float()
    if bias is not None:
        out += bias.float()
    return out.to(out_dtype)


def _to_blocked_scale(scale):
    rows, columns = scale.shape
    row_tiles = (rows + 127) // 128
    column_tiles = (columns + 3) // 4
    padded = torch.zeros(
        (row_tiles * 128, column_tiles * 4),
        device=scale.device,
        dtype=scale.dtype,
    )
    padded[:rows, :columns] = scale
    blocked = padded.view(row_tiles, 128, column_tiles, 4)
    blocked = blocked.permute(0, 2, 1, 3)
    return blocked.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _scaled_mxfp8_mm(aq, bq, sa, sb, out_dtype, block_size, bias):
    repeat = block_size // 32
    groups = (aq.shape[1] + 31) // 32
    scale_a = sa.repeat_interleave(repeat, 1)[:, :groups]
    scale_b = sb.repeat_interleave(repeat, 0)[:groups]
    scale_a = _to_blocked_scale(scale_a.to(torch.float8_e8m0fnu))
    scale_b = _to_blocked_scale(scale_b.t().contiguous().to(torch.float8_e8m0fnu))
    recipe = F.ScalingType.BlockWise1x32
    swizzle = F.SwizzleType.SWIZZLE_32_4_4
    return F.scaled_mm(
        aq,
        _column_major(bq),
        scale_a,
        recipe,
        scale_b,
        recipe,
        swizzle_a=swizzle,
        swizzle_b=swizzle,
        bias=bias,
        output_dtype=out_dtype,
    )


def _can_use_native_mxfp8(aq, bq, block_size, scale_dtype):
    return (
        aq.dtype == torch.float8_e4m3fn
        and bq.dtype == torch.float8_e4m3fn
        and scale_dtype == "fp8_e8m0"
        and block_size in (32, 64, 128)
        and torch.cuda.get_device_capability(aq.device)[0] in (10, 12)
        and getattr(torch, "float8_e8m0fnu", None) is not None
        and getattr(getattr(F, "ScalingType", None), "BlockWise1x32", None) is not None
        and getattr(getattr(F, "SwizzleType", None), "SWIZZLE_32_4_4", None) is not None
    )


@register_kernel(
    op="gemm.scaled",
    backend="torch",
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
    if aq.dtype == torch.int8 and bq.dtype == torch.int8:
        return _scaled_int8_mm(aq, bq, sa, sb, out_dtype, block_size, bias)
    if _can_use_native_mxfp8(aq, bq, block_size, scale_dtype):
        return _scaled_mxfp8_mm(aq, bq, sa, sb, out_dtype, block_size, bias)
    return _scaled_fp8_mm(aq, bq, sa, sb, out_dtype, block_size, bias)


def can_implement_scaled_grouped_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    aq, bq, sa, sb, offs, out_dtype, block_size, bias, scale_dtype = args
    result = _check_rowwise_scaling_functional("scaled_grouped_mm")
    if not result.ok:
        return result
    tensors = (aq, bq, sa, sb, offs) if bias is None else (aq, bq, sa, sb, offs, bias)
    result = first_failure(
        (
            check_same_device(tensors, "torch scaled grouped GEMM"),
            check_cuda_tensors(tensors, "torch scaled grouped GEMM"),
            check_dtypes(
                (aq, bq),
                _SCALED_GROUPED_DTYPES,
                "torch scaled grouped GEMM operands",
            ),
            check_ndim(aq, frozenset({2}), "torch scaled grouped GEMM A"),
            check_ndim(bq, frozenset({3}), "torch scaled grouped GEMM B"),
            check_ndim(sa, frozenset({2}), "torch scaled grouped GEMM A scale"),
            check_ndim(sb, frozenset({3}), "torch scaled grouped GEMM B scale"),
            check_ndim(offs, frozenset({1}), "torch scaled grouped GEMM offsets"),
            check_contiguous(offs, "torch scaled grouped GEMM offsets"),
            check_dtypes((sa, sb), _FLOAT_SCALES, "torch scaled grouped GEMM scales"),
            check_contiguous(sa, "torch scaled grouped GEMM A scale"),
            check_contiguous(sb, "torch scaled grouped GEMM B scale"),
        )
    )
    if not result.ok:
        return result
    result = first_failure(
        (
            check_dtypes(
                (offs,),
                frozenset({torch.int32}),
                "torch scaled grouped GEMM offsets",
            ),
            check_nonempty(offs, "torch scaled grouped GEMM offsets"),
        )
    )
    if not result.ok:
        return result
    if not _is_row_major(aq):
        return CheckResult(False, "torch scaled grouped GEMM requires row-major A")
    result = check_dimension_sizes_match(
        ((aq, -1), (bq, -2)),
        "torch scaled grouped GEMM contraction dimensions",
    )
    if not result.ok:
        return result
    result = check_dimension_sizes_match(
        ((bq, 0), (offs, 0)), "torch scaled grouped GEMM B group count"
    )
    if not result.ok:
        return result
    result = check_dimension_sizes_multiple_of(
        ((aq, -1), (bq, -1)), 16, "torch scaled grouped GEMM dimensions"
    )
    if not result.ok:
        return result
    result = check_values(
        (block_size,), frozenset({0}), "torch scaled grouped GEMM block_size"
    )
    if not result.ok:
        return result
    result = check_values(
        (scale_dtype,),
        frozenset({None, "fp32"}),
        "torch scaled grouped GEMM scale_dtype",
    )
    if not result.ok:
        return result
    result = first_failure(
        (
            check_shape(sa, (aq.shape[0], 1), "torch scaled grouped GEMM A scale"),
            check_shape(
                sb,
                (bq.shape[0], 1, bq.shape[2]),
                "torch scaled grouped GEMM B scale",
            ),
        )
    )
    if not result.ok:
        return result
    result = check_dtypes((out_dtype,), _BF16, "torch scaled grouped GEMM output dtype")
    if not result.ok:
        return result
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
