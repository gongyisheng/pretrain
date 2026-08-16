from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from src.kernel.registry import register_kernel
from src.kernel.spec import CheckResult
from src.kernel.utils import (
    BF16,
    FP16,
    FP32,
    FP8_E4M3,
    FP8_E8M0,
    INT32,
    check_attribute,
    check_callable,
    check_condition,
    check_compute_capability_at_least,
    check_compute_capability_in,
    check_contiguous,
    check_cuda_tensors,
    check_dimension_size_match,
    check_dimension_size_multiple_of,
    check_dtypes,
    check_matrix_layout,
    check_ndim,
    check_nonempty,
    check_same_device,
    check_shape,
    check_values,
    first_failure,
    to_column_major,
)


_SCALED_GEMM_DTYPE_PAIRS = frozenset(
    {
        (torch.int8, torch.int8),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e4m3fn),
    }
)
_OUTPUT_DTYPES = BF16 | FP16


def _is_row_major(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous() and tensor.stride(-1) == 1


def can_implement_grouped_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    a, b, offs, bias = args
    tensors = (a, b, offs) if bias is None else (a, b, offs, bias)
    checks = (
        check_same_device(tensors, "grouped GEMM"),
        check_cuda_tensors(tensors, "grouped GEMM"),
        check_dtypes((a, b), BF16, "grouped GEMM operands"),
        check_ndim(a, frozenset({2, 3}), "grouped GEMM A"),
        check_ndim(b, frozenset({2, 3}), "grouped GEMM B"),
        check_ndim(offs, frozenset({1}), "grouped GEMM offsets"),
        check_dtypes((offs,), INT32, "grouped GEMM offsets"),
        check_callable(F, "grouped_mm"),
        check_contiguous(offs, "grouped GEMM offsets"),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    checks = (
        check_matrix_layout(a, 16, "grouped GEMM A"),
        check_matrix_layout(b, 16, "grouped GEMM B"),
        check_nonempty(offs, "grouped GEMM offsets"),
        check_condition(a.ndim != 3 or b.ndim != 3, "3D x 3D not supported"),
        check_dimension_size_match(
            ((a, -1), (b, -2)), "grouped GEMM contraction dimensions"
        ),
    )
    if a.ndim == 2 and b.ndim == 3:
        checks += (
            check_dimension_size_match(
                ((b, 0), (offs, 0)), "grouped GEMM B group count"
            ),
        )
    elif a.ndim == 3:
        checks += (
            check_dimension_size_match(
                ((a, 0), (offs, 0)), "grouped GEMM A group count"
            ),
        )
    if bias is not None:
        expected_bias_shape = (offs.numel(), b.shape[-1])
        checks += (
            check_condition(a.ndim != 3, "bias not supported for ragged-N"),
            check_ndim(bias, frozenset({2}), "grouped GEMM bias"),
            check_contiguous(bias, "grouped GEMM bias"),
            check_dtypes((bias,), frozenset({a.dtype}), "grouped GEMM bias"),
            check_shape(bias, expected_bias_shape, "grouped GEMM bias"),
        )
    result = first_failure(checks)
    if not result.ok:
        return result
    return check_compute_capability_at_least(a.device, (8, 0), "grouped GEMM")


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
    checks = (
        check_same_device(tensors, feature),
        check_cuda_tensors(tensors, feature),
        check_ndim(aq, frozenset({2}), f"{feature} A"),
        check_ndim(bq, frozenset({2}), f"{feature} B"),
        check_ndim(sa, frozenset({2}), f"{feature} A scale"),
        check_ndim(sb, frozenset({2}), f"{feature} B scale"),
        check_dtypes((sa, sb), FP32, f"{feature} scales"),
        check_contiguous(sa, f"{feature} A scale"),
        check_contiguous(sb, f"{feature} B scale"),
        check_condition(block_size >= 0, "block_size must be non-negative"),
        check_condition(
            (aq.dtype, bq.dtype) in _SCALED_GEMM_DTYPE_PAIRS,
            f"{feature} does not support operand pair {aq.dtype} x {bq.dtype}",
        ),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    scale_blocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
    expected_a_scale_shape = (aq.shape[0], scale_blocks)
    expected_b_scale_shape = (scale_blocks, bq.shape[1])
    checks = (
        check_condition(_is_row_major(aq), f"{feature} requires row-major A"),
        check_condition(
            all(dimension > 0 for dimension in (aq.shape[0], aq.shape[1], bq.shape[1])),
            f"{feature} dimensions must be positive",
        ),
        check_condition(
            scale_blocks <= 1
            or (block_size >= 16 and (block_size & (block_size - 1)) == 0),
            "block_size must be a power of two >= 16",
        ),
        check_dimension_size_match(
            ((aq, -1), (bq, -2)), f"{feature} contraction dimensions"
        ),
        check_dimension_size_multiple_of(
            ((aq, -1), (bq, -1)), 16, f"{feature} dimensions"
        ),
        check_dtypes((scale_dtype,), FP32 | FP8_E8M0, f"{feature} scale_dtype"),
    )
    checks += (
        check_shape(sa, expected_a_scale_shape, f"{feature} A scale"),
        check_shape(sb, expected_b_scale_shape, f"{feature} B scale"),
        check_dtypes((out_dtype,), _OUTPUT_DTYPES, f"{feature} output dtype"),
    )
    if bias is not None:
        expected_bias_shape = (bq.shape[1],)
        checks += (
            check_dtypes((bias,), frozenset({out_dtype}), f"{feature} bias"),
            check_ndim(bias, frozenset({1}), f"{feature} bias"),
            check_contiguous(bias, f"{feature} bias"),
            check_shape(bias, expected_bias_shape, f"{feature} bias"),
        )
    return first_failure(checks)


def can_implement_scaled_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    aq, bq, sa, sb, out_dtype, block_size, bias, scale_dtype = args
    checks = (
        _scaled_common_checks(
            aq,
            bq,
            sa,
            sb,
            out_dtype,
            block_size,
            bias,
            scale_dtype,
            "scaled GEMM",
        ),
    )
    if aq.dtype == torch.int8 and bq.dtype == torch.int8:
        checks += (
            check_dimension_size_multiple_of(
                ((aq, -2),), 32, "scaled GEMM INT8 output rows"
            ),
            check_callable(torch, "_int_mm"),
        )
        minimum_capability = (8, 0)
    else:
        checks += (
            check_callable(F, "scaled_mm"),
            check_attribute(F, "ScalingType"),
        )
        minimum_capability = (8, 9)
    result = first_failure(checks)
    if not result.ok:
        return result
    if aq.dtype != torch.int8 or bq.dtype != torch.int8:
        checks += (check_attribute(F.ScalingType, "RowWise"),)
        result = first_failure(checks)
        if not result.ok:
            return result
    return check_compute_capability_at_least(
        aq.device, minimum_capability, "scaled GEMM"
    )


def _scaled_gemm_int8(aq, bq, sa, sb, out_dtype, block_size, bias):
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


def _scaled_gemm_fp8(aq, bq, sa, sb, out_dtype, block_size, bias):
    width = block_size or aq.shape[1]
    out = torch.zeros((aq.shape[0], bq.shape[1]), device=aq.device, dtype=torch.float32)
    for index, start in enumerate(range(0, aq.shape[1], width)):
        stop = min(start + width, aq.shape[1])
        partial = F.scaled_mm(
            aq[:, start:stop].contiguous(),
            to_column_major(bq[start:stop]),
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


def _scaled_gemm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias):
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
        to_column_major(bq),
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
        and scale_dtype == torch.float8_e8m0fnu
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
        return _scaled_gemm_int8(aq, bq, sa, sb, out_dtype, block_size, bias)
    if _can_use_native_mxfp8(aq, bq, block_size, scale_dtype):
        return _scaled_gemm_mxfp8(aq, bq, sa, sb, out_dtype, block_size, bias)
    return _scaled_gemm_fp8(aq, bq, sa, sb, out_dtype, block_size, bias)


def can_implement_scaled_grouped_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    aq, bq, sa, sb, offs, out_dtype, block_size, bias, scale_dtype = args
    tensors = (aq, bq, sa, sb, offs) if bias is None else (aq, bq, sa, sb, offs, bias)
    checks = (
        check_same_device(tensors, "scaled grouped GEMM"),
        check_cuda_tensors(tensors, "scaled grouped GEMM"),
        check_dtypes((aq, bq), FP8_E4M3, "scaled grouped GEMM"),
        check_ndim(aq, frozenset({2}), "scaled grouped GEMM A"),
        check_ndim(bq, frozenset({3}), "scaled grouped GEMM B"),
        check_ndim(sa, frozenset({2}), "scaled grouped GEMM A scale"),
        check_ndim(sb, frozenset({3}), "scaled grouped GEMM B scale"),
        check_ndim(offs, frozenset({1}), "scaled grouped GEMM offsets"),
        check_dtypes((sa, sb), FP32, "scaled grouped GEMM scales"),
        check_dtypes((offs,), INT32, "scaled grouped GEMM offsets"),
        check_values(
            (block_size,),
            frozenset({0}),
            "scaled grouped GEMM block_size",
        ),
        check_dtypes((scale_dtype,), FP32, "scaled grouped GEMM scale_dtype"),
    )
    checks += (
        check_callable(F, "scaled_grouped_mm"),
        check_attribute(F, "ScalingType"),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    checks += (check_attribute(F.ScalingType, "RowWise"),)
    result = first_failure(checks)
    if not result.ok:
        return result
    checks += (
        check_contiguous(offs, "scaled grouped GEMM offsets"),
        check_contiguous(sa, "scaled grouped GEMM A scale"),
        check_contiguous(sb, "scaled grouped GEMM B scale"),
        check_nonempty(offs, "scaled grouped GEMM offsets"),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    expected_a_scale_shape = (aq.shape[0], 1)
    expected_b_scale_shape = (bq.shape[0], 1, bq.shape[2])
    checks = (
        check_condition(
            _is_row_major(aq),
            "scaled grouped GEMM requires row-major A",
        ),
        check_dimension_size_match(
            ((aq, -1), (bq, -2)),
            "scaled grouped GEMM contraction dimensions",
        ),
        check_dimension_size_match(
            ((bq, 0), (offs, 0)), "scaled grouped GEMM B group count"
        ),
        check_dimension_size_multiple_of(
            ((aq, -1), (bq, -1)), 16, "scaled grouped GEMM dimensions"
        ),
        check_shape(sa, expected_a_scale_shape, "scaled grouped GEMM A scale"),
        check_shape(sb, expected_b_scale_shape, "scaled grouped GEMM B scale"),
        check_dtypes((out_dtype,), BF16, "scaled grouped GEMM output dtype"),
        check_condition(
            bias is None,
            "scaled grouped GEMM does not support bias",
        ),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    return check_compute_capability_in(
        aq.device,
        frozenset({9, 10}),
        "scaled grouped GEMM",
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
