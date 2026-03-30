import torch


def torch_moe_scatter_in(
    x_flat: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    E: int,
    capacity: int,
) -> torch.Tensor:
    """Scatter tokens into padded expert input using PyTorch indexing."""
    D = x_flat.shape[1]
    padded = x_flat.new_zeros(E, capacity, D)
    padded[expert_ids, positions] = x_flat[token_ids]
    return padded


def torch_moe_scatter_out(
    expert_out: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    weights: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Gather expert outputs and scatter-add to token positions using PyTorch."""
    gathered = expert_out[expert_ids, positions]
    weighted = gathered * weights.unsqueeze(-1)
    output = torch.zeros(T, gathered.shape[-1], device=expert_out.device, dtype=weighted.dtype)
    output.scatter_add_(0, token_ids.unsqueeze(-1).expand_as(weighted), weighted)
    return output
