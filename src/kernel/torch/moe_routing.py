import torch


def torch_moe_routing(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    n_experts: int,
    capacity_factor: float,
) -> tuple:
    """Sort-based MoE token routing (PyTorch fallback).

    Args:
        top_indices: (T, k) — expert indices per token
        top_weights: (T, k) — routing weights per token
        n_experts: total number of experts
        capacity_factor: fixed capacity = T * k * factor / E

    Returns:
        expert_ids, token_ids, weights, positions, capacity, expert_counts
    """
    T, k = top_indices.shape
    E = n_experts
    device = top_indices.device

    flat_expert_ids = top_indices.reshape(-1)
    flat_token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, k).reshape(-1)
    flat_weights = top_weights.reshape(-1)

    sorted_expert_ids, sorted_order = flat_expert_ids.sort(stable=True)
    sorted_token_ids = flat_token_ids[sorted_order]
    sorted_weights = flat_weights[sorted_order]

    expert_counts = torch.bincount(sorted_expert_ids.long(), minlength=E)
    capacity = int(T * k * capacity_factor / E)

    offsets = torch.zeros(E, dtype=torch.long, device=device)
    offsets[1:] = expert_counts[:-1].cumsum(0)
    positions = torch.arange(T * k, device=device) - offsets[sorted_expert_ids]
    keep_mask = positions < capacity

    return (
        sorted_expert_ids[keep_mask],
        sorted_token_ids[keep_mask],
        sorted_weights[keep_mask],
        positions[keep_mask],
        capacity,
        expert_counts,
    )
