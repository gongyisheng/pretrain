"""Reference quantization kernels."""

import torch

from src.kernel.registry import register_kernel


@register_kernel(
    op="quantize.pack_e2m1_rne",
    backend="eager",
    build="eager",
    autograd=False,
    capabilities=frozenset(),
    reference=True,
)
def pack_e2m1_rne(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Convert values into low-nibble-first packed E2M1 codes with RNE."""
    axis = dim % x.ndim
    magnitude = torch.nan_to_num(x.abs(), nan=6.0, posinf=6.0).clamp(max=6.0)
    codes = (
        (magnitude > 0.25).to(torch.uint8)
        + (magnitude >= 0.75).to(torch.uint8)
        + (magnitude > 1.25).to(torch.uint8)
        + (magnitude >= 1.75).to(torch.uint8)
        + (magnitude > 2.5).to(torch.uint8)
        + (magnitude >= 3.5).to(torch.uint8)
        + (magnitude > 5.0).to(torch.uint8)
    )
    codes |= (torch.signbit(x) & ~torch.isnan(x)).to(torch.uint8) << 3
    codes = codes.movedim(axis, -1).contiguous()
    packed = codes[..., 0::2] | codes[..., 1::2] << 4
    return packed.movedim(-1, axis).contiguous()
