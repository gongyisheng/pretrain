import torch

from src.kernel.ops.gemm import (
    can_implement_grouped_gemm_eager,
    can_implement_scaled_gemm_eager,
    can_implement_scaled_grouped_gemm_eager,
)
from src.kernel.registry import register_kernel


def _bounds(offs):
    ends = offs.tolist()
    return list(zip([0] + ends[:-1], ends))


@register_kernel(
    op="gemm.grouped",
    backend="eager",
    can_implement=can_implement_grouped_gemm_eager,
    build="eager",
    autograd=True,
)
def grouped_gemm(a, b, offs, bias=None):
    a_is_2d = a.ndim == 2
    b_is_2d = b.ndim == 2
    bounds = _bounds(offs)

    if not a_is_2d and not b_is_2d:
        raise NotImplementedError("3D x 3D not supported")
    if bias is not None and not a_is_2d:
        raise NotImplementedError("bias not supported for ragged-N")

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
    can_implement=can_implement_scaled_gemm_eager,
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
    scale_dtype,
    bias=None,
):
    del scale_dtype
    out = _dequant_a(aq, sa, block_size) @ _dequant_b(bq, sb, block_size)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


@register_kernel(
    op="gemm.scaled_grouped",
    backend="eager",
    can_implement=can_implement_scaled_grouped_gemm_eager,
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
    scale_dtype,
    bias=None,
):
    del scale_dtype
    a_is_2d = aq.ndim == 2
    b_is_2d = bq.ndim == 2
    bounds = _bounds(offs)

    if not a_is_2d and not b_is_2d:
        raise NotImplementedError("3D x 3D not supported")
    if bias is not None and not a_is_2d:
        raise NotImplementedError("bias not supported for ragged-N")

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
