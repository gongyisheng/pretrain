"""Numerical parity tests for activation functions vs eager mathematical refs."""
import pytest
import torch

from src.layers.activation import GATED_ACTIVATIONS, UNGATED_ACTIVATIONS
from tests.fast.layers._refs import GATED_ACTIVATIONS_REFS, UNGATED_ACTIVATIONS_REFS


DTYPES = [
    (torch.float32, 1e-5),
    (torch.float16, 5e-3),
    (torch.bfloat16, 3e-2),
]

ACT_NAMES = list(UNGATED_ACTIVATIONS.keys())


# ---------------------------- Ungated ----------------------------

@pytest.mark.parametrize("name", ACT_NAMES)
@pytest.mark.parametrize("dtype,atol", DTYPES)
def test_ungated_matches_ref(name, dtype, atol):
    """Each ungated activation matches its eager mathematical definition."""
    act = UNGATED_ACTIVATIONS[name]
    ref = UNGATED_ACTIVATIONS_REFS[name]
    x = torch.randn(2, 16, 64, dtype=dtype)
    out = act(x)
    assert out.dtype == dtype
    assert torch.allclose(out, ref(x), atol=atol)


@pytest.mark.parametrize("name", ACT_NAMES)
def test_ungated_zero_input(name):
    """All activations send 0 → 0 (relu: max(0,0); gelu: 0.5*0*(1+0)=0; silu: 0*0.5=0)."""
    act = UNGATED_ACTIVATIONS[name]
    x = torch.zeros(8, 64)
    out = act(x)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


@pytest.mark.parametrize("name", ACT_NAMES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ungated_large_positive_no_overflow(name, dtype):
    """Large positive x: all three pass through ~unchanged; output must be finite."""
    act = UNGATED_ACTIVATIONS[name]
    ref = UNGATED_ACTIVATIONS_REFS[name]
    x = torch.full((2, 16, 64), 1000.0, dtype=dtype)
    out = act(x)
    assert torch.isfinite(out).all()
    # all three behave as identity for large positive
    assert torch.allclose(out, ref(x), atol=1.0)
    assert torch.allclose(out.float(), x.float(), atol=1.0)


@pytest.mark.parametrize("name", ACT_NAMES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ungated_large_negative_saturates(name, dtype):
    """Large negative x: relu/gelu/silu all → ~0; sigmoid/exp must not underflow to nan."""
    act = UNGATED_ACTIVATIONS[name]
    x = torch.full((2, 16, 64), -1000.0, dtype=dtype)
    out = act(x)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 1e-2


# ---------------------------- Gated (GLU family) ----------------------------

@pytest.mark.parametrize("name", ACT_NAMES)
@pytest.mark.parametrize("dtype,atol", DTYPES)
def test_gated_matches_ref(name, dtype, atol):
    """Each gated activation matches act(gate) * up."""
    act = GATED_ACTIVATIONS[name]
    gate = torch.randn(2, 16, 64, dtype=dtype)
    up = torch.randn(2, 16, 64, dtype=dtype)
    out = act(gate, up)
    assert out.dtype == dtype
    # rtol scales tolerance with magnitude (output ~ act(gate)*up, can be O(10))
    rtol = 0.0 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 2e-3)
    assert torch.allclose(out, GATED_ACTIVATIONS_REFS[name](gate, up), atol=atol, rtol=rtol)


@pytest.mark.parametrize("name", ACT_NAMES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_large_input_no_overflow(name, dtype):
    """Large gate/up: output magnitude grows but must remain finite."""
    act = GATED_ACTIVATIONS[name]
    gate = torch.full((2, 16, 64), 100.0, dtype=dtype)
    up = torch.full((2, 16, 64), 100.0, dtype=dtype)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, GATED_ACTIVATIONS_REFS[name](gate, up), atol=10.0)


@pytest.mark.parametrize("name", ACT_NAMES)
def test_gated_zero_gate_zeros_output(name):
    """gate=0 → act(0)=0 → output=0 regardless of up."""
    act = GATED_ACTIVATIONS[name]
    gate = torch.zeros(2, 16, 64)
    up = torch.randn(2, 16, 64)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    assert (out == 0).all()
