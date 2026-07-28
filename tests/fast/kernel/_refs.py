import torch
import torch.nn.functional as F

_E5M2 = torch.float8_e5m2


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
    both_e5m2 = aq.dtype == _E5M2 and bq.dtype == _E5M2
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


def scaled_grouped_gemm_wgrad_ref(aq, gq, sa, sg, offs):
    """fp32 oracle for scaled grouped wgrad: gW[g] = (Xq·sa)[g]^T @ (gYq·sg)[g].

    aq (R,K), gq (R,N) fp8/int8; sa (1,K), sg (1,N) fp32 per-channel scales
    (contract axis is the ragged token dim M/dim0, so scales factor out of the
    reduction); offs (E,) int32 cumulative END-offsets. Returns (E,K,N) fp32.
    """
    a = aq.float() * sa.float()  # (R,K), sa (1,K) broadcasts over rows
    g = gq.float() * sg.float()  # (R,N)
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
    if aq.dtype in (torch.float8_e4m3fn, _E5M2) and sa.shape[-1] == 1:
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
