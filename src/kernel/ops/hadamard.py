import torch

import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.selector import dispatch


def _check_rotate(
    x: torch.Tensor,
    hadamard_block: int,
    sign_vector: torch.Tensor | None,
) -> None:
    if (
        type(hadamard_block) is not int
        or hadamard_block < 1
        or hadamard_block & (hadamard_block - 1)
    ):
        raise ValueError(
            f"hadamard_block must be a positive power of two, got {hadamard_block}"
        )
    if x.ndim < 2:
        raise ValueError(f"rotate needs a 2D or higher operand, got {x.ndim}D")
    length = x.shape[-1]
    if length % hadamard_block:
        raise ValueError(
            f"block {hadamard_block} must divide the rotated extent {length}"
        )
    if sign_vector is not None and tuple(sign_vector.shape) != (hadamard_block,):
        raise ValueError(
            f"sign_vector must hold {hadamard_block} values, "
            f"got shape {tuple(sign_vector.shape)}"
        )


def rotate(x, hadamard_block, sign_vector, inverse, out_dtype=None, backend=None):
    """Rotate `x` by a block-Hadamard along its final axis.

    The transform always runs in float32; `out_dtype` picks the stored dtype so a
    low-precision operand can land in float32 without a second pass.
    """
    _check_rotate(x, hadamard_block, sign_vector)
    if x.numel() == 0:
        return x if out_dtype is None else x.to(out_dtype)
    if sign_vector is not None and sign_vector.device != x.device:
        sign_vector = sign_vector.to(x.device)
    return dispatch(
        "hadamard.rotate",
        (x, hadamard_block, sign_vector, inverse, out_dtype),
        {},
        backend,
        device=x.device,
    )
