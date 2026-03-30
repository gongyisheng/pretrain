import torch
import torch.nn.functional as F


@torch.compile
def torch_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss using F.cross_entropy."""
    return F.cross_entropy(logits, targets)
