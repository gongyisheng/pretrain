import torch
import torch.nn as nn

from src.layers.residual import BaseResidual, StandardResidual


class BaseTransformerBlock(nn.Module):
    """Base transformer block.

    Subclasses implement attn_sublayer() and mlp_sublayer(); the residual
    strategy is selected at __init__ by passing a `residual_cls` (and
    optional `residual_kwargs`). The same class instantiates both slots —
    block.py just forwards `slot="attn"` / `slot="mlp"` plus `d_model` and
    `layer_idx` to it. Residual classes that don't care about those args
    accept them via `**_`.

    mlp_sublayer() returns (out, aux); aux is None for dense MLP and a
    load-balancing loss for MoE. Forward always returns (x, ctx, aux); ctx
    is None when using StandardResidual.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = 0,
        residual_cls: type[BaseResidual] = StandardResidual,
        residual_kwargs: dict = {},
    ):
        super().__init__()
        self.attn_res_layer = residual_cls(
            d_model, layer_idx, "attn", **residual_kwargs
        )
        self.mlp_res_layer = residual_cls(d_model, layer_idx, "mlp", **residual_kwargs)

    def attn_sublayer(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def mlp_sublayer(self, x: torch.Tensor) -> tuple:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, ctx=None, **kwargs) -> tuple:
        # Attn slot: pre → sublayer → combine.
        h = self.attn_res_layer.pre(x, ctx)
        attn_out = self.attn_sublayer(h, **kwargs)
        x, ctx = self.attn_res_layer(x, attn_out, ctx)
        # MLP slot: pre → sublayer (returns aux) → combine.
        h = self.mlp_res_layer.pre(x, ctx)
        mlp_out, aux = self.mlp_sublayer(h)
        x, ctx = self.mlp_res_layer(x, mlp_out, ctx)
        return x, ctx, aux
