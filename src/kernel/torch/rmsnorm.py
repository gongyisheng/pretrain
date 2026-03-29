import torch

@torch.compile
def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused: fp32 cast → pow2 → mean → rsqrt → scale → cast back."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x.to(dtype)