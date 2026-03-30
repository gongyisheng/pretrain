import torch
import triton
import triton.language as tl


@triton.jit
def _moe_scatter_in_kernel(
    X_ptr,          # (T, D) — flattened token embeddings
    PADDED_ptr,     # (E, C, D) — output padded input
    EXPERT_IDS_ptr, # (N,) — sorted expert ids (after capacity filter)
    TOKEN_IDS_ptr,  # (N,) — sorted token ids (after capacity filter)
    POSITIONS_ptr,  # (N,) — within-expert positions (after capacity filter)
    N,              # number of kept entries
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Scatter tokens into padded expert input buffer.

    Each program handles one (entry, d_block) tile.
    Replaces: padded_input[expert_ids, positions] = x_flat[token_ids]
    """
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

    # Load from x_flat[token_id, d_offs]
    x_ptrs = X_ptr + token_id * D + d_offs
    vals = tl.load(x_ptrs, mask=d_mask, other=0.0)

    # Store to padded[expert_id, position, d_offs]
    out_ptrs = PADDED_ptr + expert_id * C * D + position * D + d_offs
    tl.store(out_ptrs, vals, mask=d_mask)


@triton.jit
def _moe_scatter_out_kernel(
    EXPERT_OUT_ptr,  # (E, C, D) — expert output
    OUTPUT_ptr,      # (T, D) — output token embeddings (accumulated)
    EXPERT_IDS_ptr,  # (N,) — sorted expert ids
    TOKEN_IDS_ptr,   # (N,) — sorted token ids
    POSITIONS_ptr,   # (N,) — within-expert positions
    WEIGHTS_ptr,     # (N,) — routing weights
    N,               # number of kept entries
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather expert outputs and scatter-add weighted results back to token positions.

    Replaces:
        gathered = expert_out[expert_ids, positions]
        weighted = gathered * weights.unsqueeze(-1)
        output.scatter_add_(0, token_ids.unsqueeze(-1).expand_as(weighted), weighted)
    """
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

    # Load from expert_out[expert_id, position, d_offs]
    in_ptrs = EXPERT_OUT_ptr + expert_id * C * D + position * D + d_offs
    vals = tl.load(in_ptrs, mask=d_mask, other=0.0)

    # Weighted scatter-add to output[token_id, d_offs]
    weighted_vals = vals * weight
    out_ptrs = OUTPUT_ptr + token_id * D + d_offs
    tl.atomic_add(out_ptrs, weighted_vals, mask=d_mask)


def triton_moe_scatter_in(
    x_flat: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    E: int,
    capacity: int,
) -> torch.Tensor:
    """Scatter tokens into padded expert input using Triton kernel."""
    T, D = x_flat.shape
    N = expert_ids.shape[0]

    padded = torch.zeros(E, capacity, D, device=x_flat.device, dtype=x_flat.dtype)

    BLOCK_D = min(512, triton.next_power_of_2(D))
    grid = (N * triton.cdiv(D, BLOCK_D),)

    _moe_scatter_in_kernel[grid](
        x_flat, padded, expert_ids, token_ids, positions,
        N, D, capacity, BLOCK_D,
    )
    return padded


def triton_moe_scatter_out(
    expert_out: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    weights: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Gather expert outputs and scatter-add to token positions using Triton kernel."""
    E, C, D = expert_out.shape
    N = expert_ids.shape[0]

    output = torch.zeros(T, D, device=expert_out.device, dtype=expert_out.dtype)

    BLOCK_D = min(512, triton.next_power_of_2(D))
    grid = (N * triton.cdiv(D, BLOCK_D),)

    _moe_scatter_out_kernel[grid](
        expert_out, output, expert_ids, token_ids, positions, weights,
        N, D, C, BLOCK_D,
    )
    return output
