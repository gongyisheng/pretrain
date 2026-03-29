import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
    ],
    key=['seq_q', 'seq_kv', 'd_head'],
)
@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    n_heads,
    seq_q,
    seq_kv,
    d_head,
    causal,
    sm_scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Flash Attention forward kernel using online softmax.

    Each program handles one (Q block, batch, head) tile. It loads a BLOCK_Q chunk
    of queries once, then streams through K/V in BLOCK_KV tiles, maintaining a
    running softmax via three accumulators (m_i, l_i, acc) so the full [seq_q, seq_kv]
    attention matrix is never materialized. Memory usage is O(N) instead of O(N^2).

    Grid: (cdiv(seq_q, BLOCK_Q), batch * n_heads)
      - program_id(0): which Q block
      - program_id(1): which (batch, head) pair

    Args:
        Q_ptr, K_ptr, V_ptr: input tensors [batch, n_heads, seq, d_head]
        O_ptr: output tensor, same shape as Q
        L_ptr: log-sum-exp per query row [batch, n_heads, seq_q], needed for backward
        stride_*: strides for each tensor dimension (batch, head, seq, d_head)
        sm_scale: softmax scale factor, typically 1/sqrt(d_head)
        causal: if True, apply causal mask (query i can only attend to keys <= i)
            Also skips KV blocks beyond q_start + BLOCK_Q for ~2x speedup.
        BLOCK_Q: tile size along query dimension
        BLOCK_KV: tile size along key/value dimension
        BLOCK_D: tile size along head dimension (next power of 2 of d_head)
    """
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    batch_idx = bh_idx // n_heads
    head_idx = bh_idx % n_heads

    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    o_base = O_ptr + batch_idx * stride_ob + head_idx * stride_oh

    q_start = q_block_idx * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)

    q_ptrs = q_base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    q_mask = (q_offs[:, None] < seq_q) & (d_offs[None, :] < d_head)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # accumulators for online softmax
    # m_i: running row-wise max
    # l_i: running row-wise sum of exp
    # acc: running weighted sum
    m_i = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    if not causal:
        upper_bound = seq_kv
    else:
        upper_bound = tl.minimum(seq_kv, q_start + BLOCK_Q)

    for kv_block_start in range(0, upper_bound, BLOCK_KV):
        kv_offs = kv_block_start + tl.arange(0, BLOCK_KV)
        kv_mask_1d = kv_offs < seq_kv
        kv_mask = kv_mask_1d[:, None] & (d_offs[None, :] < d_head)
        k = tl.load(k_base + kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd, mask=kv_mask, other=0.0)
        score = tl.dot(q, tl.trans(k)) * sm_scale
        if not causal:
            score = tl.where(kv_offs[None, :] < seq_kv, score, float('-inf'))
        else:
            score = tl.where(q_offs[:, None] >= kv_offs[None, :], score, float('-inf'))
        # online softmax
        m_new = tl.maximum(m_i, tl.max(score, axis=1)) # new global max
        alpha = tl.exp((m_i - m_new))                  # correction factor
        p = tl.exp(score - m_new[:, None])             # new block weights (correct max)
        l_i = alpha * l_i + tl.sum(p, axis=1)          # fix old sum + add new sum
        acc = acc * alpha[:, None]                     # fix accumulation
        m_i = m_new                                    # update max

        v = tl.load(v_base + kv_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd, mask=kv_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
    
    acc = acc / l_i[:, None]
    tl.store(o_base + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od, acc, mask=q_mask)
    tl.store(L_ptr + batch_idx * (n_heads * seq_q) + head_idx * seq_q + q_offs, m_i + tl.log(l_i), mask=q_offs < seq_q)


def triton_flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal=True,
    sm_scale=None
):
    """Compute Flash Attention forward pass.

    Args:
        q: query tensor [batch, n_heads, seq_q, d_head]
        k: key tensor [batch, n_heads, seq_kv, d_head]
        v: value tensor [batch, n_heads, seq_kv, d_head]
        causal: apply causal mask (default True)
        sm_scale: softmax scale, defaults to 1/sqrt(d_head)

    Returns:
        o: attention output [batch, n_heads, seq_q, d_head]
        L: log-sum-exp per query row [batch, n_heads, seq_q], needed for backward
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch, n_heads, seq_q, d_head = q.shape
    seq_kv = k.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / (d_head ** 0.5)
    
    o = torch.empty_like(q)
    # log-sum-exp per query row, needed for backward
    L = torch.empty(batch, n_heads, seq_q, dtype=torch.float32, device=q.device)

    BLOCK_D = triton.next_power_of_2(d_head)

    grid = lambda META: (triton.cdiv(seq_q, META['BLOCK_Q']), batch * n_heads)

    _flash_attn_fwd_kernel[grid](
        q, k, v, o, L,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        n_heads, seq_q, seq_kv, d_head, causal, sm_scale,
        BLOCK_D=BLOCK_D
    )
    return o, L


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
    ],
    key=['seq_q', 'seq_kv', 'd_head'],
)
@triton.jit
def _flash_attn_dk_dv_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dkb,
    stride_dkh,
    stride_dks,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvs,
    stride_dvd,
    n_heads,
    seq_q,
    seq_kv,
    d_head,
    causal,
    sm_scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    kv_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    batch_idx = bh_idx // n_heads
    head_idx = bh_idx % n_heads

    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    do_base = dO_ptr + batch_idx * stride_dob + head_idx * stride_doh

    kv_start = kv_block_idx * BLOCK_KV
    kv_offs = kv_start + tl.arange(0, BLOCK_KV)
    d_offs = tl.arange(0, BLOCK_D)
    kv_mask = (kv_offs[:, None] < seq_kv) & (d_offs[None, :] < d_head)

    k = tl.load(k_base + kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd, mask=kv_mask, other=0.0)
    v = tl.load(v_base + kv_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd, mask=kv_mask, other=0.0)
    dk_acc = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)

    if not causal:
        lower_bound = 0
    else:
        lower_bound = kv_start
    
    for q_start in range(lower_bound, seq_q, BLOCK_Q):
        q_offs = q_start + tl.arange(0, BLOCK_Q)
        q_mask = (q_offs[:, None] < seq_q) & (d_offs[None, :] < d_head)

        q = tl.load(q_base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd, mask=q_mask, other=0.0)
        do = tl.load(do_base + q_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod, mask=q_mask, other=0.0)
        l = tl.load(L_ptr + batch_idx * (n_heads * seq_q) + head_idx * seq_q + q_offs, mask=q_offs < seq_q, other=0.0)
        d = tl.load(D_ptr + batch_idx * (n_heads * seq_q) + head_idx * seq_q + q_offs, mask=q_offs < seq_q, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(s - l[:, None])

        if causal:
            p = tl.where(q_offs[:, None] >= kv_offs[None, :], p, 0.0)
        p = tl.where((q_offs[:, None] < seq_q) & (kv_offs[None, :] < seq_kv), p, 0.0)
        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d[:, None]) * sm_scale
        dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q)
    
    dk_base = dK_ptr + batch_idx * stride_dkb + head_idx * stride_dkh
    dv_base = dV_ptr + batch_idx * stride_dvb + head_idx * stride_dvh
    tl.store(dk_base + kv_offs[:, None] * stride_dks + d_offs[None, :] * stride_dkd, dk_acc, mask=kv_mask)
    tl.store(dv_base + kv_offs[:, None] * stride_dvs + d_offs[None, :] * stride_dvd, dv_acc, mask=kv_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
    ],
    key=['seq_q', 'seq_kv', 'd_head'],
)
@triton.jit
def _flash_attn_dq_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dQ_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dqd,
    n_heads,
    seq_q,
    seq_kv,
    d_head,
    causal,
    sm_scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    batch_idx = bh_idx // n_heads
    head_idx = bh_idx % n_heads

    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    do_base = dO_ptr + batch_idx * stride_dob + head_idx * stride_doh

    q_start = q_block_idx * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)
    q_mask = (q_offs[:, None] < seq_q) & (d_offs[None, :] < d_head)

    # load Q and dO tiles (fixed for this program)
    q = tl.load(q_base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd, mask=q_mask, other=0.0)
    do = tl.load(do_base + q_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod, mask=q_mask, other=0.0)

    # load L and D per query row
    l = tl.load(L_ptr + batch_idx * (n_heads * seq_q) + head_idx * seq_q + q_offs, mask=q_offs < seq_q, other=0.0)
    d = tl.load(D_ptr + batch_idx * (n_heads * seq_q) + head_idx * seq_q + q_offs, mask=q_offs < seq_q, other=0.0)

    dq_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    if not causal:
        upper_bound = seq_kv
    else:
        upper_bound = tl.minimum(seq_kv, q_start + BLOCK_Q)

    for kv_block_start in range(0, upper_bound, BLOCK_KV):
        kv_offs = kv_block_start + tl.arange(0, BLOCK_KV)
        kv_mask = (kv_offs[:, None] < seq_kv) & (d_offs[None, :] < d_head)

        k = tl.load(k_base + kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd, mask=kv_mask, other=0.0)
        v = tl.load(v_base + kv_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd, mask=kv_mask, other=0.0)

        # recompute S and P
        s = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(s - l[:, None])

        if causal:
            p = tl.where(q_offs[:, None] >= kv_offs[None, :], p, 0.0)
        p = tl.where((q_offs[:, None] < seq_q) & (kv_offs[None, :] < seq_kv), p, 0.0)

        # dQ = dS @ K, where dS = P * (dP - D) * sm_scale
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d[:, None]) * sm_scale
        dq_acc += tl.dot(ds.to(k.dtype), k)

    # store dQ
    dq_base = dQ_ptr + batch_idx * stride_dqb + head_idx * stride_dqh
    tl.store(dq_base + q_offs[:, None] * stride_dqs + d_offs[None, :] * stride_dqd, dq_acc, mask=q_mask)


def triton_flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    L: torch.Tensor,
    do: torch.Tensor,
    causal=True,
    sm_scale=None,
):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = o.contiguous()
    do = do.contiguous()
    
    batch, n_heads, seq_q, d_head = q.shape
    seq_kv = k.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / (d_head ** 0.5)
    
    D = (do.float() * o.float()).sum(dim=-1)
    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    BLOCK_D = triton.next_power_of_2(d_head)

    grid_dkdv = lambda META: (triton.cdiv(seq_kv, META['BLOCK_KV']), batch * n_heads)
    _flash_attn_dk_dv_bwd_kernel[grid_dkdv](
        q, k, v, do, L, D, dk, dv,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        n_heads, seq_q, seq_kv, d_head, causal, sm_scale,
        BLOCK_D=BLOCK_D
    )

    grid_dq = lambda META: (triton.cdiv(seq_q, META['BLOCK_Q']), batch * n_heads)
    _flash_attn_dq_bwd_kernel[grid_dq](
        q, k, v, do, L, D, dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        n_heads, seq_q, seq_kv, d_head, causal, sm_scale,
        BLOCK_D=BLOCK_D
    )

    return dq, dk, dv


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        o, L = triton_flash_attn_fwd(q, k, v, causal=causal, sm_scale=sm_scale)
        ctx.save_for_backward(q, k, v, o, L)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        dq, dk, dv = triton_flash_attn_bwd(
            q, k, v, o, L, do,
            causal=ctx.causal,
            sm_scale=ctx.sm_scale,
        )
        return dq, dk, dv, None, None


def triton_flash_attn(q, k, v, causal=True, sm_scale=None):
    """Flash Attention with autograd support.

    Args:
        q: [batch, n_heads, seq_q, d_head]
        k: [batch, n_heads, seq_kv, d_head]
        v: [batch, n_heads, seq_kv, d_head]
        causal: apply causal mask (default True)
        sm_scale: softmax scale, defaults to 1/sqrt(d_head)

    Returns:
        o: [batch, n_heads, seq_q, d_head]
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return FlashAttention.apply(q, k, v, causal, sm_scale)