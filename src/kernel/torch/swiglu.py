import torch
import torch.nn.functional as F

@torch.compile
def torch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused: silu(gate) * up in a single pass."""
    return F.silu(gate) * up
