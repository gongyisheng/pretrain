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
