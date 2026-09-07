"""Triton block-Hadamard rotation parity against the eager correctness backend."""

import pytest
import torch

from src.kernel.backends.eager import hadamard as eager_hadamard
from src.kernel.backends.triton import hadamard as triton_hadamard
from tests.fast.helper import cuda_only


pytestmark = cuda_only


HADAMARD_BLOCKS = (2, 16, 128)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
OUT_DTYPES = (None, torch.float32, torch.float16, torch.bfloat16)
INVERSE = (False, True)
# Rotated extents are multiples of every block in HADAMARD_BLOCKS.
DENSE_SHAPES = ((256,), (256, 256), (3, 128, 384))
# Extents not divisible by the Triton tile width exercise its tail mask.
TAIL_SHAPES = ((48, 80), (2, 16, 176))
SHAPES = DENSE_SHAPES + TAIL_SHAPES
LAYOUTS = ("dense", "permuted", "strided-last-axis", "broadcast")


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("block", HADAMARD_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("inverse", INVERSE)
@pytest.mark.parametrize("with_signs", (False, True))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_rotate_precision(shape, block, dtype, out_dtype, inverse, with_signs, layout):
    if shape[-1] % block:
        pytest.skip("hadamard block must divide the rotated extent")
    torch.manual_seed(0)
    if layout == "permuted":
        source = torch.randn((2, 3, *shape), dtype=dtype, device="cuda")
        x = source.permute(1, 0, *range(2, source.ndim))
    elif layout == "strided-last-axis":
        source_shape = (*shape[:-1], shape[-1] * 2)
        x = torch.randn(source_shape, dtype=dtype, device="cuda")[..., ::2]
    elif layout == "broadcast":
        source = torch.randn((1, *shape), dtype=dtype, device="cuda")
        x = source.expand(48, *shape)
    else:
        x = torch.randn(shape, dtype=dtype, device="cuda")

    signs = None
    if with_signs:
        generator = torch.Generator().manual_seed(block)
        signs = torch.where(
            torch.rand(block, generator=generator, device="cpu") < 0.5, -1.0, 1.0
        ).to(x.device)

    actual = triton_hadamard.rotate(x, block, signs, inverse, out_dtype)
    expected = eager_hadamard.rotate(x, block, signs, inverse, out_dtype)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.dtype == (x.dtype if out_dtype is None else out_dtype)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)
