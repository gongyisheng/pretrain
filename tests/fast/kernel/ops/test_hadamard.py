"""Block-Hadamard rotation public contracts and validation."""

import pytest
import torch

import src.kernel.ops.hadamard as hadamard
from tests.fast.helper import cuda_only

EMPTY_SHAPES = ((0, 64), (8, 0), (2, 0, 64))


@cuda_only
def test_rotate_materializes_cpu_signs_before_dispatch(monkeypatch):
    x = torch.randn(3, 64, 128, device="cuda")
    signs = torch.ones(16)
    captured = {}

    def dispatch(op, args, kwargs, backend, device):
        captured.update(op=op, args=args, kwargs=kwargs, backend=backend, device=device)
        return args[0]

    monkeypatch.setattr(hadamard, "dispatch", dispatch)

    out = hadamard.rotate(x, 16, signs, False, backend="triton")

    assert out is x
    assert captured["op"] == "hadamard.rotate"
    assert captured["args"][2].device == x.device
    assert captured["backend"] == "triton"
    assert captured["device"] == x.device


@cuda_only
@pytest.mark.parametrize("backend", (None, "eager", "triton"))
def test_rotate_block_one_is_identity(backend):
    """The op handles block one before selecting a backend."""
    x = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")

    assert torch.equal(
        hadamard.rotate(x, 1, torch.ones(1, device="cuda"), False, backend=backend), x
    )
    promoted = hadamard.rotate(
        x, 1, torch.ones(1, device="cuda"), False, torch.float32, backend
    )
    assert promoted.dtype is torch.float32 and torch.equal(promoted, x.float())


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("out_dtype", (None, torch.float32))
def test_rotate_empty_operand(shape, out_dtype):
    x = torch.empty(shape, dtype=torch.bfloat16)

    out = hadamard.rotate(x, 16, None, False, out_dtype)

    assert out.shape == x.shape
    assert out.dtype == (x.dtype if out_dtype is None else out_dtype)
    assert out.numel() == 0


@pytest.mark.parametrize("block", (3, 0, -4, True, 1.5))
def test_rotate_raise_error_block(block):
    with pytest.raises(ValueError, match="power of two"):
        hadamard.rotate(torch.randn(16, 16), block, None, False)


def test_rotate_raise_error_indivisible_extent():
    with pytest.raises(ValueError, match="must divide"):
        hadamard.rotate(torch.randn(16, 24), 16, None, False)


def test_rotate_raise_error_sign_vector_length():
    with pytest.raises(ValueError, match="sign_vector"):
        hadamard.rotate(torch.randn(16, 16), 16, torch.ones(8), False)
