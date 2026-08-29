import torch

from src.kernel.registry import register_kernel
from src.kernel.utils import to_hadamard_scales


def _butterfly(v: torch.Tensor, block: int) -> torch.Tensor:
    """
    whaHadamard-transform the last axis, in place of a matmul with the Sylvester matrix.
    """
    lead = v.shape[:-1]
    for stage in range(block.bit_length() - 1):
        low = 1 << stage
        pairs = v.reshape(*lead, block // (2 * low), 2, low)
        even, odd = pairs[..., 0, :], pairs[..., 1, :]
        v = torch.stack([even + odd, even - odd], dim=-2).reshape(*lead, block)
    return v


@register_kernel(
    op="hadamard.rotate",
    backend="eager",
    build="eager",
    autograd=False,
    capabilities=frozenset(),
    reference=True,
)
def rotate(
    x: torch.Tensor,
    hadamard_block: int,
    sign_vector: torch.Tensor | None,
    inverse: bool,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-Hadamard rotation along the final axis, computed in float32."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    if hadamard_block == 1:
        return x.to(out_dtype)
    signs = None if sign_vector is None else sign_vector.to(x.device, torch.float32)
    # The +-1 matrix is unscaled, so the operand carries the 1/sqrt(block) normalization.
    pre_scale, post_scale = to_hadamard_scales(hadamard_block)

    v = x.float().reshape(*x.shape[:-1], -1, hadamard_block) * pre_scale

    if signs is not None and not inverse:
        v = v * signs
    v = _butterfly(v, hadamard_block)
    if post_scale != 1.0:
        v = v * post_scale
    if signs is not None and inverse:
        v = v * signs

    return v.reshape(x.shape).to(out_dtype)
