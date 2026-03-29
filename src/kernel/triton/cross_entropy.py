import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_fwd_kernel(
    LOGITS_ptr,
    TARGETS_ptr,
    LOSSES_ptr,
    M,
    V,
    stride_m,
    BLOCK_V: tl.constexpr,
):
    """Fused cross-entropy forward: single-pass online softmax + NLL per row.

    Each program handles one row (one token position). Iterates over vocab in
    tiles of BLOCK_V, maintaining running max and sum_exp for numerical stability.
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


@triton.jit
def _cross_entropy_bwd_kernel(
    LOGITS_ptr,
    TARGETS_ptr,
    DLOGITS_ptr,
    DLOSS_ptr,
    M,
    V,
    stride_m,
    BLOCK_V: tl.constexpr,
):
    """Fused cross-entropy backward: softmax - one_hot in single pass per row.

    First pass: compute max and sum_exp (online softmax).
    Second pass: write softmax(z) - one_hot(target), scaled by upstream gradient.
    """
    row = tl.program_id(0)
    target = tl.load(TARGETS_ptr + row)
    dloss = tl.load(DLOSS_ptr + row).to(tl.float32)

    base = row * stride_m

    # Pass 1: compute max and sum_exp
    max_val = -float('inf')
    sum_exp = 0.0
    for v_start in range(0, V, BLOCK_V):
        cols = v_start + tl.arange(0, BLOCK_V)
        mask = cols < V
        z = tl.load(LOGITS_ptr + base + cols, mask=mask, other=-float('inf')).to(tl.float32)
        chunk_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, chunk_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max

    # Pass 2: compute and write gradients
    for v_start in range(0, V, BLOCK_V):
        cols = v_start + tl.arange(0, BLOCK_V)
        mask = cols < V
        z = tl.load(LOGITS_ptr + base + cols, mask=mask, other=-float('inf')).to(tl.float32)

        # softmax(z) - one_hot(target)
        p = tl.exp(z - max_val) / sum_exp
        dz = tl.where(cols == target, p - 1.0, p) * dloss

        tl.store(DLOGITS_ptr + base + cols, dz.to(tl.bfloat16), mask=mask)


def triton_cross_entropy_fwd(logits, targets):
    """Triton fused cross-entropy forward.

    Args:
        logits: (M, V) float logits
        targets: (M,) integer targets

    Returns:
        loss: scalar mean cross-entropy loss
    """
    logits = logits.contiguous()
    M, V = logits.shape
    losses = torch.empty(M, device=logits.device, dtype=torch.float32)

    BLOCK_V = min(triton.next_power_of_2(V), 4096)
    grid = (M,)
    _cross_entropy_fwd_kernel[grid](
        logits, targets, losses,
        M, V, logits.stride(0),
        BLOCK_V=BLOCK_V,
        num_warps=8,
    )
    return losses.mean()


def triton_cross_entropy_bwd(logits, targets, dloss_expanded):
    """Triton fused cross-entropy backward.

    Args:
        logits: (M, V) float logits (saved from forward)
        targets: (M,) integer targets
        dloss_expanded: (M,) per-row upstream gradient (grad_output / M)

    Returns:
        dlogits: (M, V) gradient w.r.t. logits
    """
    logits = logits.contiguous()
    M, V = logits.shape
    dlogits = torch.empty_like(logits)

    BLOCK_V = min(triton.next_power_of_2(V), 4096)
    grid = (M,)
    _cross_entropy_bwd_kernel[grid](
        logits, targets, dlogits, dloss_expanded,
        M, V, logits.stride(0),
        BLOCK_V=BLOCK_V,
        num_warps=8,
    )
    return dlogits


class TritonCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        ctx.save_for_backward(logits, targets)
        return triton_cross_entropy_fwd(logits, targets)

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        M = logits.shape[0]
        # Expand scalar grad to per-row: grad_output / M (for mean reduction)
        dloss_expanded = torch.full((M,), grad_output.item() / M,
                                    device=logits.device, dtype=torch.float32)
        dlogits = triton_cross_entropy_bwd(logits, targets, dloss_expanded)
        return dlogits, None


def triton_cross_entropy(logits, targets):
    """Drop-in replacement for F.cross_entropy with Triton fused kernel."""
    return TritonCrossEntropy.apply(logits, targets)
