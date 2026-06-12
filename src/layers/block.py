import torch.nn as nn

from src.layers.attention import ATTN_REGISTRY
from src.layers.mlp import MLP_REGISTRY
from src.layers.norm import NORM_REGISTRY
from src.layers.residual import RESIDUAL_REGISTRY
from src.utils.config import ModelConfig


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attn slot then mlp slot.

    The residual strategy (`residual_cls`) owns each slot's combine logic; the
    block just forwards `pre → sublayer → combine` and threads `ctx` between
    slots. mlp returns (out, aux_loss); aux_loss is None for dense MLP and a
    load-balancing loss for MoE. Forward returns (x, ctx, aux_loss); ctx is None
    under StandardResidual.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        residual_cls = RESIDUAL_REGISTRY[config.residual_cls]
        self.attn_res_layer = residual_cls(
            config.d_model, layer_idx, "attn", **config.residual_kwargs
        )
        self.mlp_res_layer = residual_cls(
            config.d_model, layer_idx, "mlp", **config.residual_kwargs
        )
        norm_cls = NORM_REGISTRY[config.norm_cls]
        attn_cls = ATTN_REGISTRY[config.attn_cls]
        mlp_cls = MLP_REGISTRY[config.mlp_cls]
        self.norm1 = norm_cls(config.d_model, **config.norm_kwargs)
        self.attn = attn_cls(config.d_model, **config.attn_kwargs)
        self.norm2 = norm_cls(config.d_model, **config.norm_kwargs)
        self.mlp = mlp_cls(config.d_model, **config.mlp_kwargs)

    def forward(self, x, ctx=None, rope=None, position_ids=None, attn_mask=None):
        h = self.attn_res_layer.pre(x, ctx)
        attn_out = self.attn(
            self.norm1(h), rope=rope, position_ids=position_ids, attn_mask=attn_mask
        )
        x, ctx = self.attn_res_layer(x, attn_out, ctx)
        h = self.mlp_res_layer.pre(x, ctx)
        mlp_out, aux_loss = self.mlp(self.norm2(h))
        x, ctx = self.mlp_res_layer(x, mlp_out, ctx)
        return x, ctx, aux_loss
