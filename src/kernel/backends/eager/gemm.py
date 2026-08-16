from collections.abc import Mapping
from typing import Any

import torch

from src.kernel.registry import register_kernel
from src.kernel.spec import CheckResult
from src.kernel.utils import (
    BF16,
    FP16,
    FP32,
    FP8_E4M3,
    FP8_E5M2,
    FP8_E8M0,
    INT8,
    INT32,
    INT64,
    check_dimension_sizes_match,
    check_dtypes,
    check_ndim,
    check_nonempty,
    check_same_device,
    check_shape,
    first_failure,
)


_FLOAT_DTYPES = FP32 | BF16 | FP16
_QUANTIZED_DTYPES = INT8 | FP8_E4M3 | FP8_E5M2
_OFFSET_DTYPES = INT32 | INT64


def _check_scaled_common(
    aq: object,
    bq: object,
    sa: object,
    sb: object,
    out_dtype: object,
    block_size: object,
    bias: object,
    scale_dtype: object,
    feature: str,
) -> CheckResult:
    tensors = (aq, bq, sa, sb) if bias is None else (aq, bq, sa, sb, bias)
    result = first_failure((check_same_device(tensors, feature),))
    if not result.ok:
        return result
    if type(block_size) is not int or block_size < 0:
        return CheckResult(
            False, f"{feature} block_size must be a non-negative integer"
        )
    result = first_failure(
        (
            check_dtypes((aq, bq), _QUANTIZED_DTYPES, f"{feature} operands"),
            check_dtypes((sa, sb), FP32, f"{feature} scales"),
            check_dtypes((out_dtype,), _FLOAT_DTYPES, f"{feature} output dtype"),
            check_dtypes(
                () if scale_dtype is None else (scale_dtype,),
                FP32 | FP8_E8M0,
                f"{feature} scale_dtype",
            ),
        )
    )
    if not result.ok:
        return result
    if bias is not None:
        result = first_failure(
            (check_dtypes((bias,), _FLOAT_DTYPES, f"{feature} bias"),)
        )
        if not result.ok:
            return result
    return CheckResult(True)


def can_implement_grouped_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    if len(args) != 4:
        return CheckResult(
            False, "eager grouped GEMM requires four normalized arguments"
        )
    a, b, offs, bias = args
    tensors = (a, b, offs) if bias is None else (a, b, offs, bias)
    result = first_failure((check_same_device(tensors, "eager grouped GEMM"),))
    if not result.ok:
        return result
    result = first_failure(
        (
            check_dtypes(
                (a, b),
                _FLOAT_DTYPES,
                "eager grouped GEMM operands",
                require_same=True,
            ),
            check_ndim(a, frozenset({2, 3}), "eager grouped GEMM A"),
            check_ndim(b, frozenset({2, 3}), "eager grouped GEMM B"),
            check_ndim(offs, frozenset({1}), "eager grouped GEMM offsets"),
            check_dtypes((offs,), _OFFSET_DTYPES, "eager grouped GEMM offsets"),
            check_nonempty(offs, "eager grouped GEMM offsets"),
        )
    )
    if not result.ok:
        return result
    if a.ndim == 3 and b.ndim == 3:
        return CheckResult(False, "3D x 3D has no ragged dimension")
    result = first_failure(
        (
            check_dimension_sizes_match(
                ((a, -1), (b, -2)), "eager grouped GEMM contraction dimensions"
            ),
        )
    )
    if not result.ok:
        return result
    if a.ndim == 3:
        result = first_failure(
            (
                check_dimension_sizes_match(
                    ((a, 0), (offs, 0)), "eager grouped GEMM A group count"
                ),
            )
        )
        if not result.ok:
            return result
    if b.ndim == 3:
        result = first_failure(
            (
                check_dimension_sizes_match(
                    ((b, 0), (offs, 0)), "eager grouped GEMM B group count"
                ),
            )
        )
        if not result.ok:
            return result
    if bias is not None and a.ndim == 3:
        return CheckResult(False, "bias is not supported for the ragged-N layout")
    if bias is not None:
        result = first_failure(
            (
                check_dtypes((bias,), frozenset({a.dtype}), "eager grouped GEMM bias"),
                check_shape(
                    bias,
                    (offs.numel(), b.shape[-1]),
                    "eager grouped GEMM bias",
                ),
            )
        )
        if not result.ok:
            return result
    return CheckResult(True)


def can_implement_scaled_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    if len(args) != 8:
        return CheckResult(
            False, "eager scaled GEMM requires eight normalized arguments"
        )
    aq, bq, sa, sb, out_dtype, block_size, bias, scale_dtype = args
    result = first_failure(
        (
            _check_scaled_common(
                aq,
                bq,
                sa,
                sb,
                out_dtype,
                block_size,
                bias,
                scale_dtype,
                "eager scaled GEMM",
            ),
        )
    )
    if not result.ok:
        return result
    result = first_failure(
        (
            check_ndim(aq, frozenset({2}), "eager scaled GEMM A"),
            check_ndim(bq, frozenset({2}), "eager scaled GEMM B"),
            check_ndim(sa, frozenset({2}), "eager scaled GEMM A scale"),
            check_ndim(sb, frozenset({2}), "eager scaled GEMM B scale"),
        )
    )
    if not result.ok:
        return result
    scale_blocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
    result = first_failure(
        (
            check_dimension_sizes_match(
                ((aq, -1), (bq, -2)), "eager scaled GEMM contraction dimensions"
            ),
            check_shape(sa, (aq.shape[0], scale_blocks), "eager scaled GEMM A scale"),
            check_shape(sb, (scale_blocks, bq.shape[1]), "eager scaled GEMM B scale"),
        )
    )
    if not result.ok:
        return result
    if bias is not None:
        result = first_failure(
            (check_shape(bias, (bq.shape[1],), "eager scaled GEMM bias"),)
        )
        if not result.ok:
            return result
    return CheckResult(True)


def _bounds(offs):
    ends = offs.tolist()
    return list(zip([0] + ends[:-1], ends))


@register_kernel(
    op="gemm.grouped",
    backend="eager",
    can_implement=can_implement_grouped_gemm,
    build="eager",
    autograd=True,
)
def grouped_gemm(a, b, offs, bias=None):
    a_is_2d = a.ndim == 2
    b_is_2d = b.ndim == 2
    bounds = _bounds(offs)

    if not a_is_2d and not b_is_2d:
        raise NotImplementedError("3D x 3D has no ragged dimension; use torch.bmm")
    if bias is not None and not a_is_2d:
        raise NotImplementedError("bias is not supported for the ragged-N layout")

    if a_is_2d and not b_is_2d:
        pieces = [a[lo:hi] @ b[group] for group, (lo, hi) in enumerate(bounds)]
        out = torch.cat(pieces, dim=0)
    elif a_is_2d and b_is_2d:
        out = torch.stack([a[:, lo:hi] @ b[lo:hi] for lo, hi in bounds])
    else:
        pieces = [a[group] @ b[:, lo:hi] for group, (lo, hi) in enumerate(bounds)]
        out = torch.cat(pieces, dim=1)

    if bias is None:
        return out
    if out.ndim == 3:
        return out + bias[:, None, :]
    rows = torch.arange(out.shape[0], device=offs.device)
    return out + bias[torch.searchsorted(offs, rows, right=True)]


def _dequant_a(q, scale, block_size):
    width = block_size or q.shape[-1]
    expanded = scale.float().repeat_interleave(width, dim=-1)[..., : q.shape[-1]]
    return q.float() * expanded


def _dequant_b(q, scale, block_size):
    width = block_size or q.shape[-2]
    expanded = scale.float().repeat_interleave(width, dim=-2)[..., : q.shape[-2], :]
    return q.float() * expanded


@register_kernel(
    op="gemm.scaled",
    backend="eager",
    can_implement=can_implement_scaled_gemm,
    build="eager",
    autograd=True,
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
    del scale_dtype
    out = _dequant_a(aq, sa, block_size) @ _dequant_b(bq, sb, block_size)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def can_implement_scaled_grouped_gemm(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    if len(args) != 9:
        return CheckResult(
            False, "eager scaled grouped GEMM requires nine normalized arguments"
        )
    aq, bq, sa, sb, offs, out_dtype, block_size, bias, scale_dtype = args
    tensors = (aq, bq, sa, sb, offs) if bias is None else (aq, bq, sa, sb, offs, bias)
    result = first_failure((check_same_device(tensors, "eager scaled grouped GEMM"),))
    if not result.ok:
        return result
    result = first_failure(
        (
            _check_scaled_common(
                aq,
                bq,
                sa,
                sb,
                out_dtype,
                block_size,
                bias,
                scale_dtype,
                "eager scaled grouped GEMM",
            ),
        )
    )
    if not result.ok:
        return result
    result = first_failure(
        (
            check_ndim(aq, frozenset({2, 3}), "eager scaled grouped GEMM A"),
            check_ndim(bq, frozenset({2, 3}), "eager scaled grouped GEMM B"),
            check_ndim(offs, frozenset({1}), "eager scaled grouped GEMM offsets"),
            check_dtypes((offs,), _OFFSET_DTYPES, "eager scaled grouped GEMM offsets"),
            check_nonempty(offs, "eager scaled grouped GEMM offsets"),
        )
    )
    if not result.ok:
        return result
    if aq.ndim == 3 and bq.ndim == 3:
        return CheckResult(False, "3D x 3D has no ragged dimension")
    result = first_failure(
        (
            check_dimension_sizes_match(
                ((aq, -1), (bq, -2)),
                "eager scaled grouped GEMM contraction dimensions",
            ),
        )
    )
    if not result.ok:
        return result
    if aq.ndim == 3:
        result = first_failure(
            (
                check_dimension_sizes_match(
                    ((aq, 0), (offs, 0)),
                    "eager scaled grouped GEMM A group count",
                ),
            )
        )
        if not result.ok:
            return result
    if bq.ndim == 3:
        result = first_failure(
            (
                check_dimension_sizes_match(
                    ((bq, 0), (offs, 0)),
                    "eager scaled grouped GEMM B group count",
                ),
            )
        )
        if not result.ok:
            return result
    if aq.ndim == 2 and bq.ndim == 3:
        scale_blocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
        valid_a_scale = sa.shape == (aq.shape[0], scale_blocks)
        valid_b_scale = sb.shape == (
            bq.shape[0],
            scale_blocks,
            bq.shape[2],
        )
    elif aq.ndim == 2 and bq.ndim == 2:
        valid_a_scale = (
            sa.ndim == 2
            and sa.shape[0] == aq.shape[0]
            and (aq.shape[1] == 0 or sa.shape[1] > 0)
        )
        valid_b_scale = (
            sb.ndim == 2 and sb.shape[0] == sa.shape[1] and sb.shape[1] == bq.shape[1]
        )
        if block_size == 0:
            valid_a_scale = valid_a_scale and sa.shape[1] == offs.numel()
    else:
        scale_blocks = (aq.shape[2] + block_size - 1) // block_size if block_size else 1
        valid_a_scale = sa.shape == (
            aq.shape[0],
            aq.shape[1],
            scale_blocks,
        )
        valid_b_scale = sb.shape == (scale_blocks, bq.shape[1])
    if not valid_a_scale:
        return CheckResult(
            False, "eager scaled grouped GEMM has invalid A scale layout"
        )
    if not valid_b_scale:
        return CheckResult(
            False, "eager scaled grouped GEMM has invalid B scale layout"
        )
    if bias is not None and aq.ndim == 3:
        return CheckResult(False, "bias is not supported for the ragged-N layout")
    if bias is not None:
        result = first_failure(
            (
                check_shape(
                    bias,
                    (offs.numel(), bq.shape[-1]),
                    "eager scaled grouped GEMM bias",
                ),
            )
        )
        if not result.ok:
            return result
    return CheckResult(True)


@register_kernel(
    op="gemm.scaled_grouped",
    backend="eager",
    can_implement=can_implement_scaled_grouped_gemm,
    build="eager",
    autograd=True,
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
    del scale_dtype
    a_is_2d = aq.ndim == 2
    b_is_2d = bq.ndim == 2
    bounds = _bounds(offs)

    if not a_is_2d and not b_is_2d:
        raise NotImplementedError("3D x 3D has no ragged dimension; use torch.bmm")
    if bias is not None and not a_is_2d:
        raise NotImplementedError("bias is not supported for the ragged-N layout")

    if a_is_2d and not b_is_2d:
        a = _dequant_a(aq, sa, block_size)
        pieces = [
            a[lo:hi] @ _dequant_b(bq[group], sb[group], block_size)
            for group, (lo, hi) in enumerate(bounds)
        ]
        out = torch.cat(pieces, dim=0)
    elif a_is_2d and b_is_2d:
        pieces = []
        scale_start = 0
        for lo, hi in bounds:
            size = hi - lo
            block_count = (
                1 if block_size == 0 else (size + block_size - 1) // block_size
            )
            scale_end = scale_start + block_count
            a = _dequant_a(aq[:, lo:hi], sa[:, scale_start:scale_end], block_size)
            b = _dequant_b(bq[lo:hi], sb[scale_start:scale_end], block_size)
            pieces.append(a @ b)
            scale_start = scale_end
        out = torch.stack(pieces)
    else:
        b = _dequant_b(bq, sb, block_size)
        pieces = [
            _dequant_a(aq[group], sa[group], block_size) @ b[:, lo:hi]
            for group, (lo, hi) in enumerate(bounds)
        ]
        out = torch.cat(pieces, dim=1)

    if bias is not None:
        if out.ndim == 3:
            out = out + bias.float()[:, None, :]
        else:
            rows = torch.arange(out.shape[0], device=offs.device)
            out = out + bias.float()[torch.searchsorted(offs, rows, right=True)]
    return out.to(out_dtype)
