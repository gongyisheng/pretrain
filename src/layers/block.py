import torch
import torch.nn as nn

from src.layers.norm import LayerNorm, RMSNorm


@torch.compile(dynamic=True)
def _attn_res_aggregate_rmsnorm(
    V: torch.Tensor, w_proj: torch.Tensor, norm_weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fused: RMSNorm keys → dot-product logits → softmax → weighted sum.

    Inlining the norm eliminates the K tensor write/read to HBM.
    dynamic=True because N grows across layers.
    """
    dtype = V.dtype
    vf = V.float()
    K = (vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)).to(dtype) * norm_weight
    logits  = (K * w_proj).sum(-1)         # (N, B, S)
    weights = logits.softmax(0)            # (N, B, S)
    return (weights.unsqueeze(-1) * V).sum(0)  # (B, S, D)


@torch.compile(dynamic=True)
def _attn_res_aggregate_layernorm(
    V: torch.Tensor, w_proj: torch.Tensor,
    norm_weight: torch.Tensor, norm_bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fused: LayerNorm keys → dot-product logits → softmax → weighted sum."""
    dtype = V.dtype
    vf   = V.float()
    mean = vf.mean(-1, keepdim=True)
    K    = ((vf - mean) * torch.rsqrt((vf - mean).pow(2).mean(-1, keepdim=True) + eps)).to(dtype) * norm_weight + norm_bias
    logits  = (K * w_proj).sum(-1)
    weights = logits.softmax(0)
    return (weights.unsqueeze(-1) * V).sum(0)


def _block_attn_res(
    attn_res_ctx: list,
    x: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    """Compute block-level attention residual.

    Args:
        attn_res_ctx: list of finalized block tensors, each shape (B, S, D)
        x:   current partial block (hidden state), shape (B, S, D)
        proj: Linear(d_model, 1, bias=False) — learned query vector w_l
        norm: RMSNorm or LayerNorm applied to keys before attention

    Returns:
        Attention-weighted combination of all blocks + x, shape (B, S, D)
    """
    V = torch.stack(attn_res_ctx + [x])            # (N+1, B, S, D)
    w = proj.weight.view(-1)
    if isinstance(norm, nn.LayerNorm):
        bias = norm.bias if norm.bias is not None else torch.zeros_like(norm.weight)
        return _attn_res_aggregate_layernorm(V, w, norm.weight, bias, norm.eps)
    return _attn_res_aggregate_rmsnorm(V, w, norm.weight, norm.eps)


class BaseTransformerBlock(nn.Module):
    """Base transformer block with optional AttnRes support.

    Subclasses implement attn_sublayer() and ffn_sublayer().
    The residual logic (standard or AttnRes) lives here once for all architectures.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = 0,
        attn_res: bool = False,
        attn_res_block_size: int = 1,
        attn_res_norm: str = "rmsnorm",
    ):
        super().__init__()
        self.attn_res = attn_res
        if attn_res:
            self.layer_idx = layer_idx
            self.attn_res_block_size = attn_res_block_size
            norm_cls = LayerNorm if attn_res_norm == "layernorm" else RMSNorm
            self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
            self.attn_res_norm = norm_cls(d_model)
            self.mlp_res_proj = nn.Linear(d_model, 1, bias=False)
            self.mlp_res_norm = norm_cls(d_model)

    def attn_sublayer(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, attn_res_ctx=None, **kwargs) -> tuple:
        if self.attn_res:
            partial_block = x

            # compute h before attn using current partial_block as current-block rep
            h = _block_attn_res(attn_res_ctx, partial_block, self.attn_res_proj, self.attn_res_norm)

            # seal at block boundary before processing attn; seeds blocks with embedding at layer 0
            if self.layer_idx % self.attn_res_block_size == 0:
                attn_res_ctx = attn_res_ctx + [partial_block]
                partial_block = None

            attn_out = self.attn_sublayer(h, **kwargs)
            partial_block = partial_block + attn_out if partial_block is not None else attn_out

            # compute h before FFN
            h = _block_attn_res(attn_res_ctx, partial_block, self.mlp_res_proj, self.mlp_res_norm)

            mlp_out = self.ffn_sublayer(h)
            partial_block = partial_block + mlp_out

            return partial_block, attn_res_ctx
        else:
            # Standard residual path
            x = x + self.attn_sublayer(x, **kwargs)
            x = x + self.ffn_sublayer(x)
            return x
