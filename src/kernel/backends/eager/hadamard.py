import torch

from src.kernel.registry import register_kernel
from src.kernel.utils import to_hadamard_scales


def _butterfly(v: torch.Tensor) -> torch.Tensor:
    """
    Hadamard-transform the last axis, in place of a matmul with the Sylvester matrix.
    """
    shape = v.shape
    block = shape[-1]
    half = 1
    while half < block:
        pairs = v.reshape(-1, block // (2 * half), 2, half)
        even, odd = pairs[:, :, 0], pairs[:, :, 1]
        v = torch.stack([even + odd, even - odd], dim=2)
        half *= 2
    return v.reshape(shape)


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
    signs = None if sign_vector is None else sign_vector.to(x.device, torch.float32)
    # The +-1 matrix is unscaled, so the operand carries the 1/sqrt(block) normalization.
    pre_scale, post_scale = to_hadamard_scales(hadamard_block)

    v = x.float().reshape(*x.shape[:-1], -1, hadamard_block) * pre_scale

    if inverse:
        v = _butterfly(v)
        if signs is not None:
            v = v * signs
    else:
        if signs is not None:
            v = v * signs
        v = _butterfly(v)
    if post_scale != 1.0:
        v = v * post_scale

    return v.reshape(x.shape).to(out_dtype)
