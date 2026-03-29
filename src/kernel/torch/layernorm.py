import torch
import torch.nn.functional as F

@torch.compile
def torch_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Fused: mean → subtract → variance → normalize → scale + shift."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
