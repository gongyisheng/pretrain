import torch
import torch.nn as nn

from src.layers.activation import GATED_ACTIVATIONS, UNGATED_ACTIVATIONS
from src.layers.ffn import gated_ffn, ungated_ffn


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


class MoERouter(nn.Module):
    """MoE top-k router with fp32-pinned gate weight.

    bf16/fp16 rounding of close-competing logits before softmax can flip top-k
    picks, and low-precision storage can't accept sub-ULP gradient updates.
    Gate weight stays fp32 via _apply; forward disables autocast around the GEMM.
    """

    def __init__(self, d_model: int, n_experts: int, n_experts_per_token: int, normalize: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.normalize = normalize
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def _apply(self, fn, recurse: bool = True):
        # Pin gate.weight to fp32 across .to(bf16) / .bfloat16() / .float() / etc.
        result = super()._apply(fn, recurse)
        if self.gate.weight.dtype != torch.float32:
            self.gate.weight.data = self.gate.weight.data.float()
        return result

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        # autocast would downcast the matmul, undoing the fp32 pin.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            logits = self.gate(x.float())
            router_probs = logits.softmax(-1)
        top_weights, top_indices = torch.topk(router_probs, self.n_experts_per_token, dim=-1)
        if self.normalize:
            top_weights = top_weights / (top_weights.sum(-1, keepdim=True) + 1e-9)
        return top_indices, top_weights.to(dtype), router_probs.to(dtype)


class SparseMoEBlock(nn.Module):
    """Sparse Mixture-of-Experts FFN block with batched expert dispatch.

    Expert weights are stored as stacked tensors and processed via torch.bmm,
    replacing the sequential per-expert loop with batched GEMMs.

    Per forward pass:
      - Tokens are routed to top-k experts via MoERouter.
      - Routed tokens are sorted by expert, padded, and stacked into (E, C, D).
      - Batched matmuls compute all experts in parallel.
      - Outputs are gathered back with routing weights.
      - A load-balancing auxiliary loss (Switch Transformer formula) is returned.

    Mirrors FFN's gated/ungated split: by default `gated=True, activation="silu"`
    (SwiGLU experts); set `gated=False` for ungated experts with the same
    activation registry as FFN (relu/gelu/silu).

    Note: aux_loss scale grows linearly with n_experts_per_token (k). Under balanced
    routing the expected value is approximately k. Callers using moe_aux_loss_coef should
    account for this when comparing runs across different k values.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        n_experts: int,
        n_experts_per_token: int,
        dropout_ffn: float = 0.0,
        capacity_factor: float = None,
        bias: bool = False,
        gated: bool = True,
        activation: str = "silu",
    ):
        super().__init__()
        registry = GATED_ACTIVATIONS if gated else UNGATED_ACTIVATIONS
        if activation not in registry:
            raise ValueError(f"Unknown activation: {activation!r}; expected one of {sorted(registry)}")
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.capacity_factor = capacity_factor
        self.gated = gated
        self.act_fn = registry[activation]
        self.router = MoERouter(d_model, n_experts, n_experts_per_token)

        # Stacked expert weights: (E, out, in) following nn.Linear convention.
        # Gated path fuses gate and up into one tensor (one bmm vs two).
        if gated:
            self.expert_gate_up = nn.Parameter(torch.empty(n_experts, 2 * intermediate_size, d_model))
            self.expert_gate_up_bias = nn.Parameter(torch.zeros(n_experts, 2 * intermediate_size)) if bias else None
        else:
            self.expert_up = nn.Parameter(torch.empty(n_experts, intermediate_size, d_model))
            self.expert_up_bias = nn.Parameter(torch.zeros(n_experts, intermediate_size)) if bias else None
        self.expert_down = nn.Parameter(torch.empty(n_experts, d_model, intermediate_size))
        self.expert_down_bias = nn.Parameter(torch.zeros(n_experts, d_model)) if bias else None
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

        # --- Batched expert FFN: shared gated/ungated_ffn from layers.ffn (3D x + 3D weights) ---
        if self.gated:
            expert_out = gated_ffn(
                padded_input, self.expert_gate_up, self.expert_down, self.act_fn,
                self.expert_gate_up_bias, self.expert_down_bias,
            )
        else:
            expert_out = ungated_ffn(
                padded_input, self.expert_up, self.expert_down, self.act_fn,
                self.expert_up_bias, self.expert_down_bias,
            )
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
