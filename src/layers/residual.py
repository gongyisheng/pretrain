"""Residual strategies for transformer blocks.

`BaseResidual` defines the uniform interface; concrete strategies extend it.
TransformerBlock constructs any residual subclass via the same factory call,
so adding a new variant (gated residual, learned scalar, etc.) requires no
changes in the block.
"""

import torch
import torch.nn as nn

from src.layers.norm import LayerNorm, RMSNorm


@torch.compile(dynamic=True)
def _aggregate(V: torch.Tensor, K: torch.Tensor, w_proj: torch.Tensor) -> torch.Tensor:
    """Fused: dot-product logits → softmax → weighted sum.

    Norm is applied externally (via the optimized nn.RMSNorm/nn.LayerNorm
    wrappers from src.layers.norm) — this function only fuses the
    post-norm attention math. dynamic=True because N (number of stacked
    prior blocks) grows across layers.
    """
    logits = (K * w_proj).sum(-1)  # (N, B, S)
    weights = logits.softmax(0)  # (N, B, S)
    return (weights.unsqueeze(-1) * V).sum(0)  # (B, S, D)


class BaseResidual(nn.Module):
    """Abstract base for residual strategies used by TransformerBlock.

    Factory signature: (d_model, layer_idx, slot, **kwargs) — all three
    required, no defaults.
    Two-step API:
      pre(x, ctx) -> h         # transform x into what the sublayer sees
      forward(x, r, ctx)       # combine the base x with the sublayer's output r,
                               # returning (new_x, new_ctx)
    The sublayer call itself lives in TransformerBlock — residual only does
    residual-specific work (pre-transform and combine).

    Stores `d_model`, `layer_idx`, and `slot` as instance attributes so
    subclasses can read them via `self.*` without re-declaring them.
    `slot` ∈ {"attn", "mlp"} identifies which block slot this instance
    occupies; some strategies (e.g. AttnResidual) use it to vary behavior.
    """

    def __init__(self, d_model: int, layer_idx: int, slot: str, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.slot = slot

    def pre(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """Default: pass through. Subclasses override to transform the sublayer input."""
        return x

    def forward(self, x: torch.Tensor, r: torch.Tensor, ctx) -> tuple:
        raise NotImplementedError


class StandardResidual(BaseResidual):
    """Standard residual: out = x + r. Ctx passes through unchanged."""

    def forward(self, x: torch.Tensor, r: torch.Tensor, ctx=None):
        return x + r, ctx


class AttnResidual(BaseResidual):
    """Block-level attention residual.

    pre(x, ctx) computes h = aggregate(ctx + [x]) — a learned soft-attention
    over all prior finalized blocks plus the current partial output —
    which becomes the sublayer input.

    forward(x, r, ctx) combines: at the "attn" slot of a block-boundary
    layer (layer_idx % seal_block_size == 0), x is sealed into ctx and the
    residual base resets to 0; otherwise out = x + r. The "mlp" slot never
    seals.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        slot: str,
        *,
        norm: str = "rmsnorm",
        seal_block_size: int = 1,
        **kwargs,
    ):
        super().__init__(d_model, layer_idx, slot, **kwargs)
        norm_cls = LayerNorm if norm == "layernorm" else RMSNorm
        self.proj = nn.Linear(d_model, 1, bias=False)
        self.norm = norm_cls(d_model)
        self.seal_block_size = seal_block_size

    def pre(self, x: torch.Tensor, ctx: list[torch.Tensor]) -> torch.Tensor:
        V = torch.stack(ctx + [x])
        K = self.norm(V)
        w = self.proj.weight.view(-1)
        return _aggregate(V, K, w)

    def forward(self, x: torch.Tensor, r: torch.Tensor, ctx: list[torch.Tensor]):
        if self.slot == "attn" and (self.layer_idx % self.seal_block_size == 0):
            return r, ctx + [x]
        return x + r, ctx


# Name → class registry for YAML/config lookup.
RESIDUAL_REGISTRY: dict[str, type[BaseResidual]] = {
    "standard": StandardResidual,
    "attn_res": AttnResidual,
}
