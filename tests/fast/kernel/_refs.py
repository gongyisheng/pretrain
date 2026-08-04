import torch
import torch.nn.functional as F


def grouped_gemm_ref(a, b, offs):
    return torch._grouped_mm(a, b, offs=offs)


def grouped_gemm_wgrad_ref(a, grad_c, offs):
    return torch._grouped_mm(a.mT, grad_c, offs=offs)


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
        # fp8 rowwise/tensorwise: cuBLAS scaled matmul (col-major second operand).
        # Row-wise scaling only emits bf16/fp16, so take bf16 and widen to fp32.
        b_cm = bq.t().contiguous().t()
        out = torch._scaled_mm(
            aq, b_cm, scale_a=sa.float(), scale_b=sb.float(), out_dtype=torch.bfloat16
        )
        return out.float()
    return _dequant_a(aq, sa, ebs) @ _dequant_b(bq, sb, ebs)


def _dequant_b_grouped(bq, sb, bs):
    """Dequant per-expert B (E, K, N) with per-K-block scale (E, nkb, N)."""
    return torch.stack([_dequant_b(bq[g], sb[g], bs) for g in range(bq.shape[0])])


def _dequant_ragged(xq, scale, offs, first_block, bs):
    """Dequant (R,C) codes whose (nrb,C) scale rows tile each group's rows.

    Deliberately a plain Python loop: the oracle should restate the layout, not
    share the vectorised indexing the implementation uses.
    """
    out = torch.zeros(xq.shape, device=xq.device, dtype=torch.float32)
    bounds = [0, *offs.tolist()]
    fb = first_block.tolist()
    for gi, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        if hi <= lo:
            continue
        if bs == 0:
            out[lo:hi] = xq[lo:hi].float() * scale[fb[gi]].float()
            continue
        for j, m in enumerate(range(lo, hi, bs)):
            stop = min(m + bs, hi)
            out[m:stop] = xq[m:stop].float() * scale[fb[gi] + j].float()
    return out


def scaled_grouped_gemm_wgrad_ref(aq, gq, sa, sg, offs, first_block, bs):
    """fp32 oracle for scaled grouped wgrad: gW[g] = (Xq·sa)[g]^T @ (gYq·sg)[g].

    aq (R,K), gq (R,N) fp8/int8; sa (nrb,K), sg (nrb,N) fp32 scales whose blocks
    restart at each group, so `first_block` (E+1,) says which rows a group owns
    and `bs` is the rows per block (0 -> one block per group). offs (E,) int32
    cumulative END-offsets. Returns (E,K,N) fp32.
    """
    a = _dequant_ragged(aq, sa, offs, first_block, bs)
    g = _dequant_ragged(gq, sg, offs, first_block, bs)
    E, K, N = offs.shape[0], aq.shape[1], gq.shape[1]
    out = torch.zeros(E, K, N, device=aq.device, dtype=torch.float32)
    start = 0
    for gi, end in enumerate(offs.tolist()):
        if end > start:
            out[gi] = a[start:end].t() @ g[start:end]
        start = end
    return out


def scaled_grouped_gemm_ref(aq, bq, sa, sb, offs, ebs):
    """fp32 oracle for scaled grouped GEMM. Uses torch._scaled_grouped_mm when it
    is applicable (per-row/col fp8 on sm90/sm100); otherwise dequants and does a
    per-group matmul.

    aq (R,K), bq (E,K,N) fp8/int8; sa (R,nkb), sb (E,nkb,N) fp32 canonical scales;
    offs (E,) int32 cumulative END-offsets; ebs = elements/scale-block along K.
    """
    # Native path: per-row/col fp8 (nkb==1) on a device torch supports (sm90/sm100).
    if aq.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and sa.shape[-1] == 1:
        try:
            return torch._scaled_grouped_mm(
                aq,
                bq,
                scale_a=sa.squeeze(-1).float(),  # (R,)
                scale_b=sb.squeeze(1).float(),  # (E, N)
                offs=offs,
                out_dtype=torch.bfloat16,
            ).float()
        except (RuntimeError, ValueError):
            pass  # unsupported here (e.g. sm120) -> dequant fallback below
    a_deq = _dequant_a(aq, sa, ebs)  # (R, K) fp32
    b_deq = _dequant_b_grouped(bq, sb, ebs)  # (E, K, N) fp32
    out = torch.zeros(aq.shape[0], bq.shape[2], device=aq.device, dtype=torch.float32)
    start = 0
    for g, end in enumerate(offs.tolist()):
        if end > start:
            out[start:end] = a_deq[start:end] @ b_deq[g]
        start = end
    return out
