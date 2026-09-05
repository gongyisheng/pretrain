"""Block-Hadamard rotation op: definition, backend agreement, and validation."""

import math

import pytest
import torch

from src.kernel.ops.hadamard import rotate

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HADAMARD_BLOCKS = (2, 16, 128)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
INVERSE = (False, True)
# Shapes whose rotated extent is a multiple of every block in HADAMARD_BLOCKS.
SHAPES = ((256, 256), (3, 128, 384))
# Extents that are not multiples of 128, exercising the tail mask on both axes.
RAGGED_TAIL_SHAPES = ((48, 80), (2, 16, 176))
EMPTY_SHAPES = ((0, 64), (8, 0), (2, 0, 64))


def sylvester(block: int, device, dtype=torch.float32) -> torch.Tensor:
    """Unnormalized +-1 Sylvester matrix: entry (i, j) is (-1) ** popcount(i & j)."""
    rows = torch.arange(block, device=device)
    pairs = rows[:, None] & rows[None, :]
    parity = torch.zeros_like(pairs)
    for bit in range(block.bit_length()):
        parity = parity ^ ((pairs >> bit) & 1)
    return torch.where(parity == 0, 1.0, -1.0).to(dtype)


def oracle(x, block, signs, inverse):
    """Definition of the last-axis transform: normalize, sign, then multiply."""
    matrix = sylvester(block, x.device)
    scale = 1 / math.sqrt(block)
    blocks = x.float().reshape(*x.shape[:-1], -1, block) * scale
    if signs is not None and not inverse:
        blocks = blocks * signs
    blocks = blocks @ matrix
    if signs is not None and inverse:
        blocks = blocks * signs
    return blocks.reshape(x.shape).to(x.dtype)


def make_signs(block, device):
    generator = torch.Generator().manual_seed(block)
    draws = torch.rand(block, generator=generator, device="cpu")
    return torch.where(draws < 0.5, -1.0, 1.0).to(device)


# Worst dense-oracle error over the full grid is 3.1e-6 / 4.9e-3 / 3.9e-2; margin ~4x.
ORACLE_ATOL = {torch.float32: 1.3e-5, torch.float16: 0.021, torch.bfloat16: 0.17}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("block", HADAMARD_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inverse", INVERSE)
def test_rotate(shape, block, dtype, inverse):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    signs = make_signs(block, x.device)

    out = rotate(x, block, signs, inverse)

    assert out.shape == x.shape and out.dtype == dtype
    torch.testing.assert_close(
        out,
        oracle(x, block, signs, inverse),
        atol=ORACLE_ATOL[dtype],
        rtol=0,
    )


@pytest.mark.parametrize("block", HADAMARD_BLOCKS)
def test_rotate_without_signs(block):
    torch.manual_seed(0)
    x = torch.randn(256, 256)

    out = rotate(x, block, None, False)

    torch.testing.assert_close(
        out,
        oracle(x, block, None, False),
        atol=ORACLE_ATOL[torch.float32],
        rtol=0,
    )


@pytest.mark.parametrize("block", HADAMARD_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rotate_inverse_round_trip(block, dtype):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=dtype)
    signs = make_signs(block, x.device)

    restored = rotate(rotate(x, block, signs, False), block, signs, True)

    torch.testing.assert_close(restored, x, atol=ORACLE_ATOL[dtype], rtol=0)


@pytest.mark.parametrize("out_dtype", DTYPES)
@pytest.mark.parametrize("block", HADAMARD_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inverse", INVERSE)
def test_rotate_out_dtype(out_dtype, block, dtype, inverse):
    """`out_dtype` picks only the store dtype; the float32 transform is untouched."""
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=dtype)
    signs = make_signs(block, x.device)

    out = rotate(x, block, signs, inverse, out_dtype)

    assert out.dtype == out_dtype
    # One rounding of the float32 result, never a round trip through `x.dtype`.
    exact = rotate(x, block, signs, inverse, torch.float32)
    assert torch.equal(out, exact.to(out_dtype))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("block", HADAMARD_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", (None, torch.float32))
@pytest.mark.parametrize("inverse", INVERSE)
@CUDA
def test_rotate_backends_agree(shape, block, dtype, out_dtype, inverse):
    """Both backends run the same butterfly, so they must agree bit for bit."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    signs = make_signs(block, x.device)

    eager = rotate(x, block, signs, inverse, out_dtype, backend="eager")
    triton = rotate(x, block, signs, inverse, out_dtype, backend="triton")

    assert torch.equal(eager, triton)


@pytest.mark.parametrize("shape", RAGGED_TAIL_SHAPES)
@CUDA
def test_rotate_backends_agree_tail(shape):
    """Extents that do not fill a Triton tile must mask without disturbing the transform."""
    torch.manual_seed(0)
    block = 16
    x = torch.randn(shape, device="cuda")
    signs = make_signs(block, x.device)

    eager = rotate(x, block, signs, False, backend="eager")
    triton = rotate(x, block, signs, False, backend="triton")

    assert torch.equal(eager, triton)


@pytest.mark.parametrize("out_dtype", (None, torch.float32))
@CUDA
def test_rotate_backends_agree_permuted_leading_axes(out_dtype):
    """Every logical leading batch must address its own matrix after permutation."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 128, device="cuda").permute(1, 0, 2, 3)
    signs = make_signs(16, x.device)

    eager = rotate(x, 16, signs, False, out_dtype, backend="eager")
    triton = rotate(x, 16, signs, False, out_dtype, backend="triton")

    assert triton.shape == x.shape and triton.dtype == eager.dtype
    assert torch.equal(eager, triton)


@pytest.mark.parametrize("out_dtype", (None, torch.float32))
@CUDA
def test_rotate_backends_agree_cpu_signs_strided_last_axis(out_dtype):
    """CPU signs and a stepped final axis must use the same materialized transform."""
    torch.manual_seed(0)
    x = torch.randn(3, 64, 256, device="cuda")[..., ::2]
    signs = make_signs(16, "cpu")

    eager = rotate(x, 16, signs, False, out_dtype, backend="eager")
    triton = rotate(x, 16, signs, False, out_dtype, backend="triton")

    assert triton.shape == x.shape and triton.dtype == eager.dtype
    assert torch.equal(eager, triton)


def test_rotate_block_one_is_identity():
    """HadamardRotation treats block 1 as a no-op, so the op must agree."""
    x = torch.randn(16, 16, dtype=torch.bfloat16)

    assert torch.equal(rotate(x, 1, torch.ones(1), False), x)
    promoted = rotate(x, 1, torch.ones(1), False, torch.float32)
    assert promoted.dtype is torch.float32 and torch.equal(promoted, x.float())


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("out_dtype", (None, torch.float32))
def test_rotate_empty_operand(shape, out_dtype):
    x = torch.empty(shape, dtype=torch.bfloat16)

    out = rotate(x, 16, None, False, out_dtype)

    assert out.shape == x.shape
    assert out.dtype == (x.dtype if out_dtype is None else out_dtype)
    assert out.numel() == 0


@pytest.mark.parametrize("block", (3, 0, -4, True, 1.5))
def test_rotate_raise_error_block(block):
    with pytest.raises(ValueError, match="power of two"):
        rotate(torch.randn(16, 16), block, None, False)


def test_rotate_raise_error_indivisible_extent():
    with pytest.raises(ValueError, match="must divide"):
        rotate(torch.randn(16, 24), 16, None, False)


def test_rotate_raise_error_sign_vector_length():
    with pytest.raises(ValueError, match="sign_vector"):
        rotate(torch.randn(16, 16), 16, torch.ones(8), False)


@pytest.mark.parametrize("backend", (None, "eager", pytest.param("triton", marks=CUDA)))
def test_rotate_broadcast_operand(backend):
    """A broadcast view has no layout to store through, so it must be materialized."""
    torch.manual_seed(0)
    device = "cuda" if backend == "triton" else "cpu"
    x = torch.randn(1, 64, device=device).expand(48, 64)
    signs = make_signs(16, x.device)

    out = rotate(x, 16, signs, False, backend=backend)

    assert out.shape == x.shape
    torch.testing.assert_close(
        out, oracle(x, 16, signs, False), atol=ORACLE_ATOL[torch.float32], rtol=0
    )


def test_rotate_strided_last_axis():
    """A transpose view must rotate its logical final axis without a copy."""
    torch.manual_seed(0)
    x = torch.randn(256, 256).transpose(-2, -1)
    signs = make_signs(16, x.device)

    out = rotate(x, 16, signs, False)

    torch.testing.assert_close(
        out, oracle(x, 16, signs, False), atol=ORACLE_ATOL[torch.float32], rtol=0
    )
