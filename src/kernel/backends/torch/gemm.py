import torch

from src.kernel.registry import register_kernel
from src.kernel.spec import SupportResult


def _always_available() -> SupportResult:
    return SupportResult(True)


def _always_eligible(args, kwargs) -> SupportResult:
    return SupportResult(True)


def _add_grouped_bias(out, offs, bias):
    if bias is None:
        return out
    if out.ndim == 3:
        return out + bias[:, None, :]
    rows = torch.arange(out.shape[0], device=offs.device)
    return out + bias[torch.searchsorted(offs, rows, right=True)]


def _align_grouped_operand(x):
    align = 16 // x.dtype.itemsize
    if all(stride == 1 or stride % align == 0 for stride in x.stride()[-2:]):
        return x
    if x.stride(-2) == 1:
        leading = (x.shape[-2] + align - 1) // align * align
        matrix_stride = (1, leading)
    else:
        leading = (x.shape[-1] + align - 1) // align * align
        matrix_stride = (leading, 1)
    matrix_span = matrix_stride[0] * x.shape[-2]
    if matrix_stride[0] == 1:
        matrix_span = matrix_stride[1] * x.shape[-1]
    strides = (matrix_span, *matrix_stride) if x.ndim == 3 else matrix_stride
    aligned = torch.empty_strided(x.shape, strides, device=x.device, dtype=x.dtype)
    aligned.copy_(x)
    return aligned


@register_kernel(
    op="gemm.grouped",
    backend="torch",
    priority=-100,
    availability=_always_available,
    eligibility=_always_eligible,
    build="eager",
    autograd="native",
)
def grouped_gemm(a, b, offs, bias=None):
    if bias is not None and a.ndim == 3:
        raise NotImplementedError("bias is not supported for the ragged-N layout")
    mm_offs = offs if offs.dtype == torch.int32 else offs.to(torch.int32)
    out = torch._grouped_mm(
        _align_grouped_operand(a), _align_grouped_operand(b), offs=mm_offs
    )
    return _add_grouped_bias(out, offs, bias)


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
    backend="torch",
    priority=-100,
    availability=_always_available,
    eligibility=_always_eligible,
    build="eager",
    autograd="native",
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


def _bounds(offs):
    ends = offs.tolist()
    return list(zip([0, *ends[:-1]], ends))


@register_kernel(
    op="gemm.scaled_grouped",
    backend="torch",
    priority=-100,
    availability=_always_available,
    eligibility=_always_eligible,
    build="eager",
    autograd="native",
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
    a_is_2d, b_is_2d = aq.ndim == 2, bq.ndim == 2
    bounds = _bounds(offs)

    if not a_is_2d and not b_is_2d:
        raise NotImplementedError("3D x 3D has no ragged dim; use torch.bmm")
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
            nblocks = 1 if block_size == 0 else (size + block_size - 1) // block_size
            scale_end = scale_start + nblocks
            if size:
                a = _dequant_a(aq[:, lo:hi], sa[:, scale_start:scale_end], block_size)
                b = _dequant_b(bq[lo:hi], sb[scale_start:scale_end], block_size)
                pieces.append(a @ b)
            else:
                pieces.append(
                    torch.zeros(
                        aq.shape[0],
                        bq.shape[1],
                        device=aq.device,
                        dtype=torch.float32,
                    )
                )
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
