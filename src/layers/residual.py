import torch
import torch.nn as nn

from src.layers.norm import LayerNorm, RMSNorm


def _aggregate(V: torch.Tensor, K: torch.Tensor, w_proj: torch.Tensor) -> torch.Tensor:
    """Dot-product logits → softmax → weighted sum; norm is applied by the caller."""
    logits = (K * w_proj).sum(-1)  # (N, B, S)
    weights = logits.softmax(0)  # (N, B, S)
    return (weights.unsqueeze(-1) * V).sum(0)  # (B, S, D)


class BaseResidual(nn.Module):
    """Abstract residual strategy: pre(x, ctx) -> h, then forward(x, r, ctx) -> (x, ctx)."""

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

    @classmethod
    def compute_flops(cls, config, max_seq_len: int, layer_idx: int) -> int:
        """Forward FLOPs per token for ONE residual slot; a plain add counts as 0."""
        return 0

    @classmethod
    def compute_parameters(cls, config) -> int:
        """Trainable params for ONE residual slot. Default 0: a plain add has none."""
        return 0


class StandardResidual(BaseResidual):
    """Standard residual: out = x + r. Ctx passes through unchanged."""

    def forward(self, x: torch.Tensor, r: torch.Tensor, ctx=None):
        return x + r, ctx


class AttnResidual(BaseResidual):
    """Block-level attention residual: learned soft-attention over prior sealed blocks."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        slot: str,
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

    @classmethod
    def compute_flops(cls, config, max_seq_len: int, layer_idx: int) -> int:
        seal = config.residual_kwargs.get("seal_block_size", 1)
        n_ctx = layer_idx // seal + 1
        return 7 * n_ctx * config.d_model

    @classmethod
    def compute_parameters(cls, config) -> int:
        """proj (d_model x 1, no bias) + norm, matching __init__."""
        d = config.d_model
        norm = config.residual_kwargs.get("norm", "rmsnorm")
        norm_cls = LayerNorm if norm == "layernorm" else RMSNorm
        return d + norm_cls.compute_parameters(d)


# Name → class registry for YAML/config lookup.
RESIDUAL_REGISTRY: dict[str, type[BaseResidual]] = {
    "standard": StandardResidual,
    "attn_res": AttnResidual,
}
