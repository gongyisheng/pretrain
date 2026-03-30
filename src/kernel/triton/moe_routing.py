import torch
import triton
import triton.language as tl


@triton.jit
def _assign_positions_kernel(
    EXPERT_IDS_ptr,
    COUNTERS_ptr,
    POSITIONS_ptr,
    KEEP_MASK_ptr,
    CAPACITY,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Assign within-expert positions using atomic counters.

    Replaces the sort-based routing approach: instead of sorting T*k entries
    by expert ID, we assign each entry a position within its expert via
    atomic increment. This is O(N) instead of O(N log N).
    """
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N

    eid = tl.load(EXPERT_IDS_ptr + off, mask=mask, other=0)
    pos = tl.atomic_add(COUNTERS_ptr + eid, 1, mask=mask)
    tl.store(POSITIONS_ptr + off, pos, mask=mask)
    keep = pos < CAPACITY
    tl.store(KEEP_MASK_ptr + off, keep.to(tl.int32), mask=mask)


def triton_moe_routing(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    n_experts: int,
    capacity_factor: float,
) -> tuple:
    """Sort-free MoE token routing using atomic position assignment.

    Instead of sorting T*k routing decisions by expert ID (O(N log N)),
    assigns positions via atomic counters (O(N)). Expert IDs in the
    output are NOT sorted — the scatter kernels handle random access.

    Args:
        top_indices: (T, k) — expert indices per token
        top_weights: (T, k) — routing weights per token
        n_experts: total number of experts
        capacity_factor: fixed capacity = T * k * factor / E

    Returns:
        expert_ids: (N_kept,) — expert ids (not sorted)
        token_ids: (N_kept,) — token ids
        weights: (N_kept,) — routing weights
        positions: (N_kept,) — within-expert positions
        capacity: int
        expert_counts: (E,) — atomic counters (actual expert loads)
    """
    T, k = top_indices.shape
    E = n_experts
    N = T * k
    device = top_indices.device
    capacity = int(N * capacity_factor / E)

    flat_expert_ids = top_indices.reshape(-1)
    flat_token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, k).reshape(-1)
    flat_weights = top_weights.reshape(-1)

    counters = torch.zeros(E, dtype=torch.long, device=device)
    positions = torch.empty(N, dtype=torch.long, device=device)
    keep_mask = torch.empty(N, dtype=torch.int32, device=device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _assign_positions_kernel[grid](
        flat_expert_ids, counters, positions, keep_mask,
        capacity, N, BLOCK_SIZE,
    )

    km = keep_mask.bool()
    return (
        flat_expert_ids[km],
        flat_token_ids[km],
        flat_weights[km],
        positions[km],
        capacity,
        counters,
    )
