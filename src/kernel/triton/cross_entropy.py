import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_fwd_kernel(
    LOGITS_ptr,
    TARGETS_ptr,
    LOSSES_ptr,
    MAX_ptr,
    SUMEXP_ptr,
    M,
    V,
    stride_m,
    BLOCK_V: tl.constexpr,
):
    """Fused cross-entropy forward: single-pass online softmax + NLL per row.

    Each program handles one row (one token position). Iterates over vocab in
    tiles of BLOCK_V, maintaining running max and sum_exp for numerical stability.
    Also stores max and sum_exp per row for use in backward (avoids recomputation).
    """
    row = tl.program_id(0)
    target = tl.load(TARGETS_ptr + row)

    max_val = -float('inf')
    sum_exp = 0.0
    target_logit = 0.0

    base = row * stride_m
    for v_start in range(0, V, BLOCK_V):
        cols = v_start + tl.arange(0, BLOCK_V)
        mask = cols < V
        z = tl.load(LOGITS_ptr + base + cols, mask=mask, other=-float('inf')).to(tl.float32)

        # Online softmax update
        chunk_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, chunk_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max

        # Gather target logit
        target_logit += tl.sum(tl.where(cols == target, z, 0.0))

    loss = -target_logit + max_val + tl.log(sum_exp)
    tl.store(LOSSES_ptr + row, loss)
    tl.store(MAX_ptr + row, max_val)
    tl.store(SUMEXP_ptr + row, sum_exp)


@triton.jit
def _cross_entropy_bwd_kernel(
    LOGITS_ptr,
    TARGETS_ptr,
    DLOGITS_ptr,
    DLOSS_ptr,
    MAX_ptr,
    SUMEXP_ptr,
    M,
    V,
    stride_m,
    BLOCK_V: tl.constexpr,
):
    """Fused cross-entropy backward using saved max/sum_exp from forward.

    Single pass: compute softmax(z) - one_hot(target), scaled by upstream gradient.
    """
    row = tl.program_id(0)
    target = tl.load(TARGETS_ptr + row)
    dloss = tl.load(DLOSS_ptr + row).to(tl.float32)
    max_val = tl.load(MAX_ptr + row)
    sum_exp = tl.load(SUMEXP_ptr + row)

    base = row * stride_m

    for v_start in range(0, V, BLOCK_V):
        cols = v_start + tl.arange(0, BLOCK_V)
        mask = cols < V
        z = tl.load(LOGITS_ptr + base + cols, mask=mask, other=-float('inf')).to(tl.float32)

        # softmax(z) - one_hot(target)
        p = tl.exp(z - max_val) / sum_exp
        dz = tl.where(cols == target, p - 1.0, p) * dloss

        tl.store(DLOGITS_ptr + base + cols, dz, mask=mask)


def triton_cross_entropy_fwd(logits, targets):
    """Triton fused cross-entropy forward.

    Args:
        logits: (M, V) float logits
        targets: (M,) integer targets

    Returns:
        (loss, max_logits, sum_exp): scalar loss + saved stats for backward
    """
    logits = logits.contiguous()
    M, V = logits.shape
    losses = torch.empty(M, device=logits.device, dtype=torch.float32)
    max_logits = torch.empty(M, device=logits.device, dtype=torch.float32)
    sum_exp = torch.empty(M, device=logits.device, dtype=torch.float32)

    BLOCK_V = min(triton.next_power_of_2(V), 4096)
    grid = (M,)
    _cross_entropy_fwd_kernel[grid](
        logits, targets, losses, max_logits, sum_exp,
        M, V, logits.stride(0),
        BLOCK_V=BLOCK_V,
        num_warps=8,
    )
    return losses.mean(), max_logits, sum_exp


def triton_cross_entropy_bwd(logits, targets, dloss_expanded, max_logits, sum_exp):
    """Triton fused cross-entropy backward using saved forward stats.

    Args:
        logits: (M, V) float logits (saved from forward)
        targets: (M,) integer targets
        dloss_expanded: (M,) per-row upstream gradient
        max_logits: (M,) saved max per row from forward
        sum_exp: (M,) saved sum_exp per row from forward

    Returns:
        dlogits: (M, V) gradient w.r.t. logits
    """
    logits = logits.contiguous()
    M, V = logits.shape
    dlogits = torch.empty_like(logits)

    BLOCK_V = min(triton.next_power_of_2(V), 4096)
    grid = (M,)
    _cross_entropy_bwd_kernel[grid](
        logits, targets, dlogits, dloss_expanded, max_logits, sum_exp,
        M, V, logits.stride(0),
        BLOCK_V=BLOCK_V,
        num_warps=8,
    )
    return dlogits


class TritonCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        loss, max_logits, sum_exp = triton_cross_entropy_fwd(logits, targets)
        ctx.save_for_backward(logits, targets, max_logits, sum_exp)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, max_logits, sum_exp = ctx.saved_tensors
        M = logits.shape[0]
        # Expand scalar grad to per-row without .item() sync
        dloss_expanded = grad_output.expand(M).contiguous() / M
        dlogits = triton_cross_entropy_bwd(logits, targets, dloss_expanded, max_logits, sum_exp)
        return dlogits, None


def triton_cross_entropy(logits, targets):
    """Drop-in replacement for F.cross_entropy with Triton fused kernel."""
    return TritonCrossEntropy.apply(logits, targets)
