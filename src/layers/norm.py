import torch
import torch.nn as nn


@torch.compile
def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused: fp32 cast → pow2 → mean → rsqrt → scale → cast back."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        out = _rmsnorm(x.reshape(-1, orig_shape[-1]), self.weight, self.eps)
        return out.reshape(orig_shape)
