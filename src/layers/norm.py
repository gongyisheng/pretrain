import torch
import torch.nn as nn


class RMSNorm(nn.RMSNorm):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__(d_model, eps=eps, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.to(self.weight.dtype)).to(x.dtype)


class LayerNorm(nn.LayerNorm):
    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True):
        super().__init__(d_model, eps=eps, elementwise_affine=True, bias=bias)


NORM_REGISTRY = {
    "rmsnorm": RMSNorm,
    "layernorm": LayerNorm,
}
