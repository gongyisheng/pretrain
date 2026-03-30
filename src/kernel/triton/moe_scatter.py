import torch
import triton
import triton.language as tl


@triton.jit
def _moe_scatter_in_kernel(
    X_ptr,          # (T, D) — flattened token embeddings
    PADDED_ptr,     # (E, C, D) — output padded input
    EXPERT_IDS_ptr, # (N,) — expert ids (after capacity filter)
    TOKEN_IDS_ptr,  # (N,) — token ids (after capacity filter)
    POSITIONS_ptr,  # (N,) — within-expert positions (after capacity filter)
    N,              # number of kept entries
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Scatter tokens into padded expert input buffer."""
    pid = tl.program_id(0)
    entry_id = pid // tl.cdiv(D, BLOCK_D)
    d_block = (pid % tl.cdiv(D, BLOCK_D)) * BLOCK_D

    if entry_id >= N:
        return

    expert_id = tl.load(EXPERT_IDS_ptr + entry_id)
    token_id = tl.load(TOKEN_IDS_ptr + entry_id)
    position = tl.load(POSITIONS_ptr + entry_id)

    d_offs = d_block + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    x_ptrs = X_ptr + token_id * D + d_offs
    vals = tl.load(x_ptrs, mask=d_mask, other=0.0)

    out_ptrs = PADDED_ptr + expert_id * C * D + position * D + d_offs
    tl.store(out_ptrs, vals, mask=d_mask)


@triton.jit
def _moe_scatter_in_bwd_kernel(
    GRAD_PADDED_ptr, # (E, C, D) — gradient of padded input
    GRAD_X_ptr,      # (T, D) — gradient of x_flat (accumulated)
    EXPERT_IDS_ptr,  # (N,)
    TOKEN_IDS_ptr,   # (N,)
    POSITIONS_ptr,   # (N,)
    N,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward for scatter-in: gather grad from padded, accumulate into grad_x."""
    pid = tl.program_id(0)
    entry_id = pid // tl.cdiv(D, BLOCK_D)
    d_block = (pid % tl.cdiv(D, BLOCK_D)) * BLOCK_D

    if entry_id >= N:
        return

    expert_id = tl.load(EXPERT_IDS_ptr + entry_id)
    token_id = tl.load(TOKEN_IDS_ptr + entry_id)
    position = tl.load(POSITIONS_ptr + entry_id)

    d_offs = d_block + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    grad_ptrs = GRAD_PADDED_ptr + expert_id * C * D + position * D + d_offs
    grad_vals = tl.load(grad_ptrs, mask=d_mask, other=0.0)

    # Atomic add: a token routed to k experts accumulates gradients from all k
    out_ptrs = GRAD_X_ptr + token_id * D + d_offs
    tl.atomic_add(out_ptrs, grad_vals, mask=d_mask)


@triton.jit
def _moe_scatter_out_kernel(
    EXPERT_OUT_ptr,  # (E, C, D) — expert output
    OUTPUT_ptr,      # (T, D) — output token embeddings (accumulated)
    EXPERT_IDS_ptr,  # (N,)
    TOKEN_IDS_ptr,   # (N,)
    POSITIONS_ptr,   # (N,)
    WEIGHTS_ptr,     # (N,)
    N,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather expert outputs and scatter-add weighted results back."""
    pid = tl.program_id(0)
    entry_id = pid // tl.cdiv(D, BLOCK_D)
    d_block = (pid % tl.cdiv(D, BLOCK_D)) * BLOCK_D

    if entry_id >= N:
        return

    expert_id = tl.load(EXPERT_IDS_ptr + entry_id)
    token_id = tl.load(TOKEN_IDS_ptr + entry_id)
    position = tl.load(POSITIONS_ptr + entry_id)
    weight = tl.load(WEIGHTS_ptr + entry_id)

    d_offs = d_block + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    in_ptrs = EXPERT_OUT_ptr + expert_id * C * D + position * D + d_offs
    vals = tl.load(in_ptrs, mask=d_mask, other=0.0)

    weighted_vals = vals * weight
    out_ptrs = OUTPUT_ptr + token_id * D + d_offs
    tl.atomic_add(out_ptrs, weighted_vals, mask=d_mask)


@triton.jit
def _moe_scatter_out_bwd_expert_kernel(
    GRAD_OUTPUT_ptr, # (T, D)
    GRAD_EXPERT_ptr, # (E, C, D)
    EXPERT_IDS_ptr,
    TOKEN_IDS_ptr,
    POSITIONS_ptr,
    WEIGHTS_ptr,
    N,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward for scatter-out w.r.t. expert_out: grad_expert[e,p] = grad_output[t] * w."""
    pid = tl.program_id(0)
    entry_id = pid // tl.cdiv(D, BLOCK_D)
    d_block = (pid % tl.cdiv(D, BLOCK_D)) * BLOCK_D

    if entry_id >= N:
        return

    expert_id = tl.load(EXPERT_IDS_ptr + entry_id)
    token_id = tl.load(TOKEN_IDS_ptr + entry_id)
    position = tl.load(POSITIONS_ptr + entry_id)
    weight = tl.load(WEIGHTS_ptr + entry_id)

    d_offs = d_block + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    grad_ptrs = GRAD_OUTPUT_ptr + token_id * D + d_offs
    grad_vals = tl.load(grad_ptrs, mask=d_mask, other=0.0)

    out_ptrs = GRAD_EXPERT_ptr + expert_id * C * D + position * D + d_offs
    tl.store(out_ptrs, grad_vals * weight, mask=d_mask)


# --- Autograd wrapper for scatter-in ---

def _run_scatter_in(x_flat, expert_ids, token_ids, positions, E, capacity):
    T, D = x_flat.shape
    N = expert_ids.shape[0]
    padded = torch.zeros(E, capacity, D, device=x_flat.device, dtype=x_flat.dtype)
    BLOCK_D = min(512, triton.next_power_of_2(D))
    grid = (N * triton.cdiv(D, BLOCK_D),)
    _moe_scatter_in_kernel[grid](x_flat, padded, expert_ids, token_ids, positions, N, D, capacity, BLOCK_D)
    return padded


def _run_scatter_in_bwd(grad_padded, expert_ids, token_ids, positions, T):
    E, C, D = grad_padded.shape
    N = expert_ids.shape[0]
    grad_x = torch.zeros(T, D, device=grad_padded.device, dtype=grad_padded.dtype)
    BLOCK_D = min(512, triton.next_power_of_2(D))
    grid = (N * triton.cdiv(D, BLOCK_D),)
    _moe_scatter_in_bwd_kernel[grid](grad_padded, grad_x, expert_ids, token_ids, positions, N, D, C, BLOCK_D)
    return grad_x


class TritonMoEScatterIn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, expert_ids, token_ids, positions, E, capacity):
        ctx.save_for_backward(expert_ids, token_ids, positions)
        ctx.T = x_flat.shape[0]
        return _run_scatter_in(x_flat, expert_ids, token_ids, positions, E, capacity)

    @staticmethod
    def backward(ctx, grad_padded):
        expert_ids, token_ids, positions = ctx.saved_tensors
        grad_x = _run_scatter_in_bwd(grad_padded, expert_ids, token_ids, positions, ctx.T)
        return grad_x, None, None, None, None, None


def triton_moe_scatter_in(x_flat, expert_ids, token_ids, positions, E, capacity):
    return TritonMoEScatterIn.apply(x_flat, expert_ids, token_ids, positions, E, capacity)


# --- Autograd wrapper for scatter-out ---

def _run_scatter_out(expert_out, expert_ids, token_ids, positions, weights, T):
    E, C, D = expert_out.shape
    N = expert_ids.shape[0]
    output = torch.zeros(T, D, device=expert_out.device, dtype=expert_out.dtype)
    BLOCK_D = min(512, triton.next_power_of_2(D))
    grid = (N * triton.cdiv(D, BLOCK_D),)
    _moe_scatter_out_kernel[grid](expert_out, output, expert_ids, token_ids, positions, weights, N, D, C, BLOCK_D)
    return output


def _run_scatter_out_bwd_expert(grad_output, expert_ids, token_ids, positions, weights, E, C):
    D = grad_output.shape[1]
    N = expert_ids.shape[0]
    grad_expert = torch.zeros(E, C, D, device=grad_output.device, dtype=grad_output.dtype)
    BLOCK_D = min(512, triton.next_power_of_2(D))
    grid = (N * triton.cdiv(D, BLOCK_D),)
    _moe_scatter_out_bwd_expert_kernel[grid](grad_output, grad_expert, expert_ids, token_ids, positions, weights, N, D, C, BLOCK_D)
    return grad_expert


class TritonMoEScatterOut(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_out, expert_ids, token_ids, positions, weights, T):
        # Only save what's needed for backward (not expert_out — saves memory)
        ctx.save_for_backward(expert_ids, token_ids, positions, weights)
        ctx.T = T
        ctx.E, ctx.C, ctx.D = expert_out.shape
        return _run_scatter_out(expert_out, expert_ids, token_ids, positions, weights, T)

    @staticmethod
    def backward(ctx, grad_output):
        expert_ids, token_ids, positions, weights = ctx.saved_tensors
        grad_expert = _run_scatter_out_bwd_expert(grad_output, expert_ids, token_ids, positions, weights, ctx.E, ctx.C)
        return grad_expert, None, None, None, None, None


def triton_moe_scatter_out(expert_out, expert_ids, token_ids, positions, weights, T):
    return TritonMoEScatterOut.apply(expert_out, expert_ids, token_ids, positions, weights, T)
