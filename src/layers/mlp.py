import torch
import torch.nn as nn

from src.layers.activation import GATED_ACTIVATIONS, UNGATED_ACTIVATIONS


def gated_mlp(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    act_fn,
    b_gate_up: torch.Tensor = None,
    b_down: torch.Tensor = None,
) -> torch.Tensor:
    """Fused gated MLP: gate_up matmul → chunk → act(gate, up) → down matmul.

    Shapes: x (..., D); w_gate_up (2*I, D); w_down (D, I); 1D biases. Leading
    dims broadcast through `@`.
    """
    gate_up = x @ w_gate_up.mT
    if b_gate_up is not None:
        gate_up = gate_up + (
            b_gate_up if b_gate_up.ndim == 1 else b_gate_up.unsqueeze(-2)
        )
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = act_fn(gate, up)
    out = hidden @ w_down.mT
    if b_down is not None:
        out = out + (b_down if b_down.ndim == 1 else b_down.unsqueeze(-2))
    return out


def ungated_mlp(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    act_fn,
    b_up: torch.Tensor = None,
    b_down: torch.Tensor = None,
) -> torch.Tensor:
    """Fused ungated MLP: up matmul → act(up) → down matmul.

    Shapes (parallel to gated_mlp): x (..., D); w_up (I, D); w_down (D, I).
    """
    up = x @ w_up.mT
    if b_up is not None:
        up = up + (b_up if b_up.ndim == 1 else b_up.unsqueeze(-2))
    hidden = act_fn(up)
    out = hidden @ w_down.mT
    if b_down is not None:
        out = out + (b_down if b_down.ndim == 1 else b_down.unsqueeze(-2))
    return out


def grouped_mlp(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_down: torch.Tensor,
    act_fn,
    offs: torch.Tensor,
    gated: bool,
    row_expert_ids: torch.Tensor = None,
    b_in: torch.Tensor = None,
    b_down: torch.Tensor = None,
) -> torch.Tensor:
    """Dropless MoE expert FFN over expert-sorted tokens via grouped GEMM.

    x is (R, D) with rows grouped by expert and `offs` the (E,) int32 cumulative
    end-offsets (offs[-1] == R). Weights keep nn.Linear (E, out, in) layout; the
    transposed view w.mT is the (E, in, out) form torch._grouped_mm expects.
    Empty groups (count 0) are handled by torch._grouped_mm. Bias, when given,
    is added per row using row_expert_ids.

    Eager supports all dtypes; under torch.compile only bf16 is supported
    (torch._grouped_mm meta limitation). Under autocast, operands are cast to
    the autocast dtype (e.g. bf16) so grouped GEMM receives the expected dtype,
    mirroring what autocast does for ordinary matmuls.
    """
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        dt = torch.get_autocast_dtype(dev)
        x = x.to(dt)
        w_in = w_in.to(dt)
        w_down = w_down.to(dt)
        if b_in is not None:
            b_in = b_in.to(dt)
        if b_down is not None:
            b_down = b_down.to(dt)
    h = torch._grouped_mm(x, w_in.mT, offs=offs)
    if b_in is not None:
        h = h + b_in[row_expert_ids]
    if gated:
        gate, up = h.chunk(2, dim=-1)
        h = act_fn(gate, up)
    else:
        h = act_fn(h)
    out = torch._grouped_mm(h, w_down.mT, offs=offs)
    if b_down is not None:
        out = out + b_down[row_expert_ids]
    return out


class DenseMLPBlock(nn.Module):
    """Configurable dense feed-forward block.

    Two structural variants selected by `gated`:
      - gated=False: down_proj(activation(up_proj(x)))
      - gated=True (GLU family): down_proj(activation(gate, up)) where
        (gate, up) = chunk(gate_up_proj(x), 2, dim=-1)

    `activation` is one of ``"relu"`` / ``"gelu"`` / ``"silu"``. Combinations:
        gated=True + activation="silu" → SwiGLU
        gated=True + activation="gelu" → GeGLU
        gated=True + activation="relu" → ReGLU

    Bias defaults to False (modern LLM convention); pass `bias=True` for the
    GPT-2 / classic Transformer convention.

    Returns a tuple (out, None) to match the uniform MLP block contract.
    """

    def __init__(
        self,
        d_model: int,
        *,
        intermediate_size: int,
        activation: str = "silu",
        gated: bool = True,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Defaults/validation (intermediate_size, activation) live in ModelConfig.
        registry = GATED_ACTIVATIONS if gated else UNGATED_ACTIVATIONS
        self.gated = gated
        self.act_fn = registry[activation]
        if gated:
            self.gate_up_proj = nn.Linear(d_model, 2 * intermediate_size, bias=bias)
        else:
            self.up_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        if self.gated:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            hidden = self.act_fn(gate, up)
        else:
            hidden = self.act_fn(self.up_proj(x))
        return self.dropout(self.down_proj(hidden)), None

    @classmethod
    def compute_flops(cls, d_model, *, intermediate_size, gated=True, bias=False, **_):
        d_ff = intermediate_size
        if gated:
            matmul = 6 * d_model * d_ff
            b = (2 * d_ff + d_model) if bias else 0
        else:
            matmul = 4 * d_model * d_ff
            b = (d_ff + d_model) if bias else 0
        return matmul + b

    @classmethod
    def compute_parameters(
        cls, d_model, *, intermediate_size, gated=True, bias=False, active=False, **_
    ) -> int:
        d_ff = intermediate_size
        if gated:
            weights = 3 * d_ff * d_model  # gate_up (2*d_ff x d) + down (d x d_ff)
            b = (2 * d_ff + d_model) if bias else 0
        else:
            weights = 2 * d_ff * d_model  # up (d_ff x d) + down (d x d_ff)
            b = (d_ff + d_model) if bias else 0
        return weights + b


# ---------------------------------------------------------------------------
# MoE routing helpers
# ---------------------------------------------------------------------------


class ExpertBias(nn.Module):
    """Auxiliary-loss-free load-balancing bias (arXiv:2408.15664).

    A per-expert bias added to gating scores for top-k selection only (it never
    enters the combine weights). It is not a learned parameter — `update` nudges
    it toward uniform load with a fixed-rate sign rule. The buffer is fp32-pinned
    via _apply so the small update steps survive bf16/fp16 model casts.
    """

    def __init__(self, n_experts: int, update_rate: float = 0.001):
        super().__init__()
        self.update_rate = update_rate
        self.register_buffer("bias", torch.zeros(n_experts))

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        if self.bias.dtype != torch.float32:
            self.bias = self.bias.float()
        return result

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return scores + self.bias

    @torch.no_grad()
    def update(self, expert_counts: torch.Tensor):
        err = expert_counts.float().mean() - expert_counts.float()
        self.bias += self.update_rate * err.sign()

    @classmethod
    def compute_flops(cls, n_experts: int) -> int:
        return n_experts  # add the bias to each expert score

    @classmethod
    def compute_parameters(cls, n_experts: int) -> int:
        return n_experts  # the bias buffer


class ExpertLoad(nn.Module):
    def __init__(self, n_experts: int):
        super().__init__()
        self.register_buffer(
            "train_load", torch.zeros(n_experts, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "eval_load", torch.zeros(n_experts, dtype=torch.long), persistent=False
        )

    def record_load(self, counts: torch.Tensor, training: bool) -> None:
        if training:
            self.train_load += counts
        else:
            self.eval_load.copy_(counts)

    def reset_train_load(self) -> None:
        self.train_load.zero_()

    def reset_eval_load(self) -> None:
        self.eval_load.zero_()


MOE_ROUTER_SCORE_FNS = {
    "softmax": lambda logits: logits.softmax(dim=-1),
    "sigmoid": lambda logits: logits.sigmoid(),
}


class MoERouter(nn.Module):
    """MoE top-k router with fp32-pinned gate weight.

    bf16/fp16 rounding of close-competing logits before softmax can flip top-k
    picks, and low-precision storage can't accept sub-ULP gradient updates.
    Gate weight stays fp32 via _apply; forward disables autocast around the GEMM.

    With `expert_bias`, an independent `ExpertBias` submodule shifts the gating
    scores for selection only (combine weights stay the original softmax probs).
    """

    def __init__(
        self,
        d_model: int,
        n_routed_experts: int,
        n_routed_experts_per_token: int,
        normalize: bool = True,
        expert_bias: bool = False,
        expert_bias_update_rate: float = 0.001,
        router_score_fn: str = "sigmoid",
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_routed_experts_per_token = n_routed_experts_per_token
        self.normalize = normalize
        self.score_fn = MOE_ROUTER_SCORE_FNS[router_score_fn]
        self.gate = nn.Linear(d_model, n_routed_experts, bias=False)
        self.expert_bias = (
            ExpertBias(n_routed_experts, expert_bias_update_rate)
            if expert_bias
            else None
        )

    def _apply(self, fn, recurse: bool = True):
        # Pin gate.weight to fp32
        result = super()._apply(fn, recurse)
        if self.gate.weight.dtype != torch.float32:
            self.gate.weight.data = self.gate.weight.data.float()
        return result

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        k = self.n_routed_experts_per_token
        # autocast would downcast the matmul, undoing the fp32 pin.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            logits = self.gate(x.float())
            router_probs = self.score_fn(logits)
        if self.expert_bias is not None:
            top_indices = torch.topk(self.expert_bias(router_probs), k, dim=-1).indices
            top_weights = router_probs.gather(-1, top_indices)
        else:
            top_weights, top_indices = torch.topk(router_probs, k, dim=-1)
        if self.normalize:
            top_weights = top_weights / (top_weights.sum(-1, keepdim=True) + 1e-9)
        return top_indices, top_weights.to(dtype), router_probs.to(dtype)

    def update_expert_bias(self, expert_counts: torch.Tensor):
        """Nudge the load-balancing bias toward uniform load. No-op if disabled."""
        if self.expert_bias is not None:
            self.expert_bias.update(expert_counts)

    @classmethod
    def compute_flops(cls, d_model: int, n_experts: int, *, expert_bias=False) -> int:
        flops = 2 * d_model * n_experts  # gate matmul (no bias)
        if expert_bias:
            flops += ExpertBias.compute_flops(n_experts)
        return flops

    @classmethod
    def compute_parameters(cls, d_model: int, n_experts: int, *, expert_bias=False):
        params = d_model * n_experts  # gate, no bias
        if expert_bias:
            params += ExpertBias.compute_parameters(n_experts)
        return params


class SparseMoEBlock(nn.Module):
    """Sparse Mixture-of-Experts MLP block (dropless dispatch).

    Expert weights are stored as stacked (E, out, in) tensors. Tokens are sorted
    by expert and run through a variable-group GEMM (`grouped_mlp` /
    torch._grouped_mm) with no padding and no dropped tokens.

    Per forward pass:
      - Tokens are routed to top-k experts via MoERouter.
      - Experts run via the path above; outputs are gathered back with routing weights.
      - A load-balancing auxiliary loss (Switch Transformer formula) is returned.

    Mirrors DenseMLPBlock's gated/ungated split: by default `gated=True, activation="silu"`
    (SwiGLU experts); set `gated=False` for ungated experts.

    Note: aux_loss scale grows linearly with n_routed_experts_per_token (k). Under balanced
    routing the expected value is approximately k. Callers using aux_loss_coef should
    account for this when comparing runs across different k values.
    """

    def __init__(
        self,
        d_model: int,
        *,
        intermediate_size: int,
        n_routed_experts: int,
        n_routed_experts_per_token: int,
        n_shared_experts: int = 0,
        aux_loss: bool = True,
        aux_loss_coef: float = 0.001,
        expert_bias: bool = False,
        expert_bias_update_rate: float = 0.001,
        router_score_fn: str = "sigmoid",
        activation: str = "silu",
        gated: bool = True,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if aux_loss and expert_bias:
            raise ValueError(
                "aux_loss and expert_bias are mutually exclusive MoE balancing strategies"
            )
        registry = GATED_ACTIVATIONS if gated else UNGATED_ACTIVATIONS
        self.n_routed_experts = n_routed_experts
        self.n_routed_experts_per_token = n_routed_experts_per_token
        self.n_shared_experts = n_shared_experts
        self.aux_loss = aux_loss
        self.aux_loss_coef = aux_loss_coef
        self.router_score_fn = router_score_fn
        self.expert_bias = expert_bias
        self.expert_bias_update_rate = expert_bias_update_rate
        self.gated = gated
        self.act_fn = registry[activation]
        self.router = MoERouter(
            d_model,
            n_routed_experts,
            n_routed_experts_per_token,
            expert_bias=expert_bias,
            expert_bias_update_rate=expert_bias_update_rate,
            router_score_fn=router_score_fn,
        )

        # DeepSeekMoE shared experts: always-on FFN run on every token, merged
        # into one dense block of width n_shared_experts * intermediate_size and
        # added to the routed output. Not part of routing or the aux loss.
        if n_shared_experts > 0:
            self.shared_expert = DenseMLPBlock(
                d_model,
                intermediate_size=n_shared_experts * intermediate_size,
                activation=activation,
                gated=gated,
                bias=bias,
                dropout=dropout,
            )
        else:
            self.shared_expert = None

        # Stacked expert weights: (E, out, in) following nn.Linear convention.
        # Gated path fuses gate and up into one tensor (one bmm vs two).
        if gated:
            self.expert_gate_up = nn.Parameter(
                torch.empty(n_routed_experts, 2 * intermediate_size, d_model)
            )
            self.expert_gate_up_bias = (
                nn.Parameter(torch.zeros(n_routed_experts, 2 * intermediate_size))
                if bias
                else None
            )
        else:
            self.expert_up = nn.Parameter(
                torch.empty(n_routed_experts, intermediate_size, d_model)
            )
            self.expert_up_bias = (
                nn.Parameter(torch.zeros(n_routed_experts, intermediate_size))
                if bias
                else None
            )
        self.expert_down = nn.Parameter(
            torch.empty(n_routed_experts, d_model, intermediate_size)
        )
        self.expert_down_bias = (
            nn.Parameter(torch.zeros(n_routed_experts, d_model)) if bias else None
        )
        self.expert_dropout = nn.Dropout(dropout)

        self.expert_load = ExpertLoad(n_routed_experts)

    def forward(self, x: torch.Tensor):
        # x: (B, S, D)
        B, S, D = x.shape
        BS = B * S
        k = self.n_routed_experts_per_token
        E = self.n_routed_experts
        x_flat = x.view(BS, D)

        # top_indices: (BS, k), top_weights: (BS, k), router_probs: (BS, E)
        top_indices, top_weights, router_probs = self.router(x_flat)

        w_in = self.expert_gate_up if self.gated else self.expert_up
        b_in = self.expert_gate_up_bias if self.gated else self.expert_up_bias

        # flatten
        expert_ids = top_indices.reshape(-1)
        token_ids = torch.arange(BS, device=x.device).repeat_interleave(k)
        weights = top_weights.reshape(-1, 1)

        # sort
        expert_ids_sorted, order = expert_ids.sort(stable=True)
        token_ids_sorted = token_ids[order]
        weights_sorted = weights[order]
        expert_counts = torch.bincount(expert_ids_sorted, minlength=E)
        self.expert_load.record_load(expert_counts.detach(), self.training)
        offs = expert_counts.cumsum(0).to(torch.int32)
        x_sorted = x_flat[token_ids_sorted]

        # grouped mm
        expert_out = grouped_mlp(
            x_sorted,
            w_in,
            self.expert_down,
            self.act_fn,
            offs,
            self.gated,
            row_expert_ids=expert_ids_sorted if b_in is not None else None,
            b_in=b_in,
            b_down=self.expert_down_bias,
        )
        expert_out = expert_out * weights_sorted
        expert_out = self.expert_dropout(expert_out)

        # unsort, sum
        output = torch.zeros_like(x_flat)
        output.index_add_(0, token_ids_sorted, expert_out)

        aux_loss = None
        if self.aux_loss:
            with torch.no_grad():
                f = expert_counts.to(x.dtype) / BS  # (E,)
            probs = router_probs
            # normalize sigmoid token probs, softmax no-op
            if self.router_score_fn == "sigmoid":
                probs = probs / (probs.sum(-1, keepdim=True) + 1e-9)
            P = probs.mean(0)  # (E,)
            aux_loss = E * (f * P).sum()

        output = output.view(B, S, D)
        if self.shared_expert is not None:
            shared_out, _ = self.shared_expert(x)
            output = output + shared_out
        return output, aux_loss

    @torch.no_grad()
    def post_step(self):
        if self.expert_bias:
            self.router.update_expert_bias(self.expert_load.train_load)
            self.expert_load.reset_train_load()

    def forward_meta(self) -> dict:
        """Routing metadata from the last forward. `expert_load` is the accumulated
        train counts in train mode, the current batch's counts in eval mode."""
        load = (
            self.expert_load.train_load if self.training else self.expert_load.eval_load
        )
        return {"expert_load": load}

    @classmethod
    def compute_flops(
        cls,
        d_model,
        *,
        intermediate_size,
        n_routed_experts,
        n_routed_experts_per_token=2,
        n_shared_experts=0,
        gated=True,
        bias=False,
        expert_bias=False,
        **_,
    ):
        d_ff = intermediate_size
        router = MoERouter.compute_flops(
            d_model, n_routed_experts, expert_bias=expert_bias
        )
        if gated:
            expert = 6 * d_model * d_ff
            b = (2 * d_ff + d_model) if bias else 0
        else:
            expert = 4 * d_model * d_ff
            b = (d_ff + d_model) if bias else 0
        shared = (
            DenseMLPBlock.compute_flops(
                d_model,
                intermediate_size=n_shared_experts * d_ff,
                gated=gated,
                bias=bias,
            )
            if n_shared_experts > 0
            else 0
        )
        return router + n_routed_experts_per_token * (expert + b) + shared

    @classmethod
    def compute_parameters(
        cls,
        d_model,
        *,
        intermediate_size,
        n_routed_experts,
        n_routed_experts_per_token=2,
        n_shared_experts=0,
        gated=True,
        bias=False,
        expert_bias=False,
        active=False,
        **_,
    ) -> int:
        d_ff = intermediate_size
        router = MoERouter.compute_parameters(
            d_model, n_routed_experts, expert_bias=expert_bias
        )
        # active counts only the k experts that actually run per token.
        n = n_routed_experts_per_token if active else n_routed_experts
        if gated:
            expert = 3 * d_ff * d_model
            b = (2 * d_ff + d_model) if bias else 0
        else:
            expert = 2 * d_ff * d_model
            b = (d_ff + d_model) if bias else 0
        # Shared experts always run, so they count toward both active and total.
        shared = (
            DenseMLPBlock.compute_parameters(
                d_model,
                intermediate_size=n_shared_experts * d_ff,
                gated=gated,
                bias=bias,
            )
            if n_shared_experts > 0
            else 0
        )
        return router + n * (expert + b) + shared


MLP_REGISTRY = {"dense": DenseMLPBlock, "moe": SparseMoEBlock}
