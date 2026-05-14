import torch
import torch.nn as nn
import torch.nn.functional as F


def _route(
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
    n_experts: int,
    capacity_factor: float,
) -> tuple:
    """Sort-based MoE token routing with fixed capacity (drops overflow tokens).

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


def _scatter_in(
    x_flat: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    E: int,
    capacity: int,
) -> torch.Tensor:
    """Scatter tokens into padded expert input (E, capacity, D)."""
    D = x_flat.shape[1]
    padded = x_flat.new_zeros(E, capacity, D)
    padded[expert_ids, positions] = x_flat[token_ids]
    return padded


def _scatter_out(
    expert_out: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    weights: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Gather expert outputs and scatter-add to token positions."""
    gathered = expert_out[expert_ids, positions]
    weighted = gathered * weights.unsqueeze(-1)
    output = torch.zeros(T, gathered.shape[-1], device=expert_out.device, dtype=weighted.dtype)
    output.scatter_add_(0, token_ids.unsqueeze(-1).expand_as(weighted), weighted)
    return output


@torch.compile
def _expert_ffn(
    padded_input: torch.Tensor,
    expert_gate_up: torch.Tensor,
    expert_down: torch.Tensor,
    expert_gate_up_bias: torch.Tensor = None,
    expert_down_bias: torch.Tensor = None,
) -> torch.Tensor:
    """Fused batched expert FFN: gate_up bmm → chunk → SwiGLU → down bmm.

    Args:
        padded_input: (E, C, D) — padded token embeddings per expert
        expert_gate_up: (E, 2*I, D) — stacked gate+up projection weights
        expert_down: (E, D, I) — stacked down projection weights
        expert_gate_up_bias: (E, 2*I) — optional bias
        expert_down_bias: (E, D) — optional bias

    Returns:
        (E, C, D) — expert output
    """
    gate_up = torch.bmm(padded_input, expert_gate_up.mT)  # (E, C, 2*I)
    if expert_gate_up_bias is not None:
        gate_up = gate_up + expert_gate_up_bias.unsqueeze(1)
    gate, up = gate_up.chunk(2, dim=-1)                     # each (E, C, I)
    hidden = F.silu(gate) * up                              # (E, C, I)
    out = torch.bmm(hidden, expert_down.mT)                 # (E, C, D)
    if expert_down_bias is not None:
        out = out + expert_down_bias.unsqueeze(1)
    return out


class MoERouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, n_experts_per_token: int, normalize: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.normalize = normalize
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (T, d_model)  where T = B*S (flattened)
        logits = self.gate(x)                                        # (T, n_experts)
        router_probs = logits.softmax(-1)                            # (T, n_experts)
        top_weights, top_indices = torch.topk(router_probs, self.n_experts_per_token, dim=-1)
        if self.normalize:
            top_weights = top_weights / (top_weights.sum(-1, keepdim=True) + 1e-9)
        return top_indices, top_weights, router_probs


class SparseMoEBlock(nn.Module):
    """Sparse Mixture-of-Experts FFN block with batched expert dispatch.

    Expert weights are stored as stacked tensors and processed via torch.bmm,
    replacing the sequential per-expert loop with 3 batched GEMMs.

    For each forward pass:
      - Tokens are routed to top-k experts via MoERouter.
      - Routed tokens are sorted by expert, padded, and stacked into (E, C, D).
      - Batched matmuls compute all experts in parallel.
      - Outputs are gathered back with routing weights.
      - A load-balancing auxiliary loss (Switch Transformer formula) is returned.

    Note: aux_loss scale grows linearly with n_experts_per_token (k). Under balanced
    routing the expected value is approximately k. Callers using moe_aux_loss_coef should
    account for this when comparing runs across different k values.
    """

    def __init__(self, d_model: int, intermediate_size: int, n_experts: int, n_experts_per_token: int, dropout_ffn: float = 0.0, capacity_factor: float = None, bias: bool = False):
        super().__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.capacity_factor = capacity_factor
        self.router = MoERouter(d_model, n_experts, n_experts_per_token)

        # Stacked expert weights: (E, out, in) following nn.Linear convention.
        # gate and up are fused into one tensor to save memory (one bmm instead of two)
        self.expert_gate_up = nn.Parameter(torch.empty(n_experts, 2 * intermediate_size, d_model))
        self.expert_down = nn.Parameter(torch.empty(n_experts, d_model, intermediate_size))
        if bias:
            self.expert_gate_up_bias = nn.Parameter(torch.zeros(n_experts, 2 * intermediate_size))
            self.expert_down_bias = nn.Parameter(torch.zeros(n_experts, d_model))
        else:
            self.expert_gate_up_bias = None
            self.expert_down_bias = None
        self.expert_dropout = nn.Dropout(dropout_ffn)

    def forward(self, x: torch.Tensor):
        # x: (B, S, D)
        B, S, D = x.shape
        T = B * S
        k = self.n_experts_per_token
        E = self.n_experts
        x_flat = x.view(T, D)

        top_indices, top_weights, router_probs = self.router(x_flat)
        # top_indices: (T, k)   top_weights: (T, k)   router_probs: (T, E)

        if self.capacity_factor is not None:
            # --- Optimized routing with capacity filtering ---
            sorted_expert_ids, sorted_token_ids, sorted_weights, positions, capacity, expert_counts = (
                _route(top_indices, top_weights, E, self.capacity_factor)
            )
        else:
            # --- Dynamic capacity: no tokens dropped ---
            flat_expert_ids = top_indices.reshape(-1)
            flat_token_ids = torch.arange(T, device=x.device).unsqueeze(1).expand(T, k).reshape(-1)
            flat_weights = top_weights.reshape(-1)
            sorted_expert_ids, sorted_order = flat_expert_ids.sort(stable=True)
            sorted_token_ids = flat_token_ids[sorted_order]
            sorted_weights = flat_weights[sorted_order]
            expert_counts = torch.bincount(sorted_expert_ids.long(), minlength=E)
            capacity = (expert_counts.max().item() + 31) // 32 * 32
            offsets = torch.zeros(E, dtype=torch.long, device=x.device)
            offsets[1:] = expert_counts[:-1].cumsum(0)
            positions = torch.arange(len(sorted_expert_ids), device=x.device) - offsets[sorted_expert_ids]

        # --- Build padded input: (E, capacity, D) ---
        padded_input = _scatter_in(x_flat, sorted_expert_ids, sorted_token_ids, positions, E, capacity)

        # --- Batched expert FFN: fused gate+up bmm, then down bmm ---
        expert_out = _expert_ffn(padded_input, self.expert_gate_up, self.expert_down, self.expert_gate_up_bias, self.expert_down_bias)
        expert_out = self.expert_dropout(expert_out)

        # --- Scatter results back with routing weights ---
        output = _scatter_out(expert_out, sorted_expert_ids, sorted_token_ids, positions, sorted_weights, T)

        # --- Switch Transformer load-balancing auxiliary loss ---
        with torch.no_grad():
            # Vectorized: f_i = fraction of tokens routed to expert i
            f = expert_counts.to(x.dtype) / T                   # (E,)

        P = router_probs.mean(0)                                 # (E,)
        aux_loss = E * (f * P).sum()

        return output.view(B, S, D), aux_loss
