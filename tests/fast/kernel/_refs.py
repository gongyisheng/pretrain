import torch
import torch.nn.functional as F


def _add_bias(out, offs, bias):
    """Add a (G,N) bias to a grouped output, broadcast over its row dim.

    Per group slice when the output is 3D (ragged K), else per output row -- every
    row belongs to exactly one group. Promotes to fp32, so an oracle keeps summing
    in fp32 whatever dtype the GEMM under test returned.
    """
    if bias is None:
        return out
    out = out.float()
    if out.ndim == 3:
        return out + bias.float()[:, None, :]
    rows = torch.arange(out.shape[0], device=out.device)
    return out + bias.float()[torch.searchsorted(offs, rows, right=True)]


def grouped_gemm_ref(a, b, offs, bias=None):
    """torch._grouped_mm, plus the optional (G,N) bias that torch itself rejects.

    Without a bias the torch dtype passes through untouched; with one the sum is
    formed in fp32.
    """
    return _add_bias(torch._grouped_mm(a, b, offs=offs), offs, bias)


def _dequant_a(q, scale, bs):
    """Dequant A-operand (M, K) with per-K-block scale (M, nkb) along the last axis."""
    qf = q.float()
    M, K = qf.shape
    nkb = scale.shape[-1]
    pad = nkb * bs - K
    if pad:
        qf = F.pad(qf, (0, pad))
    deq = (qf.reshape(M, nkb, bs) * scale.float().unsqueeze(-1)).reshape(M, nkb * bs)
    return deq[:, :K]


def _dequant_b(q, scale, bs):
    """Dequant B-operand (K, N) with per-K-block scale (nkb, N) along dim 0."""
    qf = q.float()
    K, N = qf.shape
    nkb = scale.shape[0]
    pad = nkb * bs - K
    if pad:
        qf = F.pad(qf, (0, 0, 0, pad))
    deq = (qf.reshape(nkb, bs, N) * scale.float().unsqueeze(1)).reshape(nkb * bs, N)
    return deq[:K, :]


def scaled_gemm_ref(aq, bq, sa, sb, ebs):
    factorable = sa.shape[-1] == 1 and sb.shape[0] == 1
    both_e5m2 = aq.dtype == torch.float8_e5m2 and bq.dtype == torch.float8_e5m2
    if factorable and not both_e5m2:
        if aq.dtype == torch.int8:
            acc = torch._int_mm(aq, bq)  # int32 (M, N)
            return acc.float() * sa.float() * sb.float()
        # fp8: unit scales, so cuBLAS accumulates raw codes and the scales come out
        # after as in the int8 branch -- CUDA 13 dropped row-wise fp8 on sm120.
        one = torch.ones((1, 1), device=aq.device)
        b_cm = bq.t().contiguous().t()
        acc = torch._scaled_mm(
            aq, b_cm, scale_a=one, scale_b=one, out_dtype=torch.float32
        )
        return acc * sa.float() * sb.float()
    return _dequant_a(aq, sa, ebs) @ _dequant_b(bq, sb, ebs)


def _dequant_ragged(xq, scale, offs, bs):
    """Dequant (R,C) codes whose (nrb,C) scale rows tile each group's rows.

    Deliberately a plain Python loop that walks the scale rows with its own running
    counter: the oracle should restate the layout, not share the block bookkeeping
    the implementation and the kernel derive.
    """
    out = torch.zeros(xq.shape, device=xq.device, dtype=torch.float32)
    bounds = [0, *offs.tolist()]
    # blocks restart at each group, so a group's rows start where the last one's ended
    first_block = 0
    for lo, hi in zip(bounds, bounds[1:]):
        if bs == 0:
            if hi > lo:
                out[lo:hi] = xq[lo:hi].float() * scale[first_block].float()
            first_block += 1  # an empty group still owns its one row
            continue
        for j, m in enumerate(range(lo, hi, bs)):
            stop = min(m + bs, hi)
            out[m:stop] = xq[m:stop].float() * scale[first_block + j].float()
        first_block += (hi - lo + bs - 1) // bs
    return out


def scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, block_size, bias=None):
    """fp32 oracle for the scaled grouped GEMM, all three ragged layouts.

    Layout follows (aq.ndim, bq.ndim), as in the kernel. `sa`/`sb` each mirror their
    operand's axis order with the contraction axis replaced by the scale-block axis.
    `block_size` 0 means one block spanning the whole contraction segment. The
    optional (E,N) `bias` is added after the scales, as the kernel's epilogue does.

    Deliberately restates each layout with its own loop rather than sharing the
    kernel's block bookkeeping.
    """
    a_2d, b_2d = aq.ndim == 2, bq.ndim == 2
    bounds = list(zip([0, *offs.tolist()], offs.tolist()))
    E = offs.shape[0]
    if bias is not None and not a_2d:
        raise NotImplementedError("bias is not supported for the ragged-N layout")

    if a_2d and not b_2d:  # ragged M: (R,K) x (E,K,N) -> (R,N)
        K = aq.shape[1]
        a = _dequant_a(aq, sa, block_size or K)
        out = torch.zeros(
            aq.shape[0], bq.shape[2], device=aq.device, dtype=torch.float32
        )
        for g, (lo, hi) in enumerate(bounds):
            if hi > lo:
                out[lo:hi] = a[lo:hi] @ _dequant_b(bq[g], sb[g], block_size or K)
    elif a_2d and b_2d:  # ragged K: (M,R) x (R,N) -> (E,M,N)
        a = _dequant_ragged(aq.mT, sa.mT, offs, block_size).mT  # (M,R)
        b = _dequant_ragged(bq, sb, offs, block_size)  # (R,N)
        out = torch.zeros(
            E, aq.shape[0], bq.shape[1], device=aq.device, dtype=torch.float32
        )
        for g, (lo, hi) in enumerate(bounds):
            if hi > lo:
                out[g] = a[:, lo:hi] @ b[lo:hi]
    else:  # ragged N: (E,M,K) x (K,R) -> (M,R)
        K = aq.shape[2]
        b = _dequant_b(bq, sb, block_size or K)  # (K,R), blocks along K
        out = torch.zeros(
            aq.shape[1], bq.shape[1], device=aq.device, dtype=torch.float32
        )
        for g, (lo, hi) in enumerate(bounds):
            if hi > lo:
                out[:, lo:hi] = _dequant_a(aq[g], sa[g], block_size or K) @ b[:, lo:hi]
    return _add_bias(out, offs, bias)
