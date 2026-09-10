"""Backend-neutral quantization operations."""

import torch

import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.selector import dispatch


_SUPPORTED_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})


def pack_e2m1_rne(
    x: torch.Tensor, dim: int = -1, backend: str | None = None
) -> torch.Tensor:
    """Convert values into low-nibble-first packed E2M1 codes with RNE."""
    if dim not in (-2, -1):
        raise ValueError(f"dim must be -2 or -1, got {dim}")
    if x.ndim < -dim:
        raise ValueError(f"dim {dim} is invalid for a {x.ndim}D tensor")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"x must have dtype float32, float16, or bfloat16, got {x.dtype}"
        )
    if x.shape[dim] % 2:
        raise ValueError(
            f"fp4_e2m1 contraction extent must be even, got {x.shape[dim]}"
        )
    return dispatch("quantize.pack_e2m1_rne", (x, dim), {}, backend, device=x.device)
