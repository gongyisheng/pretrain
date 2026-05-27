"""Numerical parity tests for activation functions vs eager mathematical refs."""

import pytest
import torch

from src.layers.activation import GATED_ACTIVATIONS, UNGATED_ACTIVATIONS
from tests.fast.layers._refs import (
    GATED_ACTIVATIONS_REFS,
    SIMPLE_DTYPES,
    UNGATED_ACTIVATIONS_REFS,
)


DTYPES = SIMPLE_DTYPES
ACT_NAMES = list(UNGATED_ACTIVATIONS.keys())
GATED_ACT_NAMES = list(GATED_ACTIVATIONS.keys())
# Sub-groups for tests that need different input magnitudes or behavior assumptions.
SIMPLE_UNGATED = ["relu", "gelu", "silu", "leaky_relu"]
SQUARED_UNGATED = ["relu2", "gelu2", "silu2", "leaky_relu2"]
# Ungated names whose output saturates to ~0 for large negative x.
# leaky variants are excluded — they preserve a small slope on the negative side.
SATURATING_UNGATED = ["relu", "gelu", "silu", "relu2", "gelu2", "silu2"]
SIMPLE_GATED = ["relu", "gelu", "silu", "leaky_relu", "bilinear", "powlu"]
SQUARED_GATED = ["relu2", "gelu2", "silu2", "leaky_relu2", "bilinear2"]


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
    # rtol scales with output magnitude — squared variants can reach ~10 in fp16,
    # and 1 ULP at that scale is ~0.01 > the atol tuned for output magnitudes ~1.
    rtol = (
        0.0 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 2e-3)
    )
    assert torch.allclose(out, ref(x), atol=atol, rtol=rtol)


@pytest.mark.parametrize("name", ACT_NAMES)
def test_ungated_zero_input(name):
    """All activations send 0 → 0 (relu: max(0,0); gelu: 0.5*0*(1+0)=0; silu: 0*0.5=0)."""
    act = UNGATED_ACTIVATIONS[name]
    x = torch.zeros(8, 64)
    out = act(x)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


@pytest.mark.parametrize("name", SIMPLE_UNGATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ungated_large_positive_no_overflow_simple(name, dtype):
    """Large positive x: simple unary acts pass through ~unchanged; output must be finite."""
    act = UNGATED_ACTIVATIONS[name]
    ref = UNGATED_ACTIVATIONS_REFS[name]
    x = torch.full((2, 16, 64), 1000.0, dtype=dtype)
    out = act(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref(x), atol=1.0)


@pytest.mark.parametrize("name", SQUARED_UNGATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ungated_large_positive_no_overflow_squared(name, dtype):
    """Large positive x for squared variants: 200² = 4e4 fits fp16 (max 65504)."""
    act = UNGATED_ACTIVATIONS[name]
    ref = UNGATED_ACTIVATIONS_REFS[name]
    x = torch.full((2, 16, 64), 200.0, dtype=dtype)
    out = act(x)
    assert torch.isfinite(out).all()
    # Squared output is ~4e4; atol scales with magnitude.
    assert torch.allclose(out, ref(x), atol=100.0)


@pytest.mark.parametrize("name", SATURATING_UNGATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ungated_large_negative_saturates(name, dtype):
    """Large negative x: relu/gelu/silu all → ~0; sigmoid/exp must not underflow to nan."""
    act = UNGATED_ACTIVATIONS[name]
    x = torch.full((2, 16, 64), -1000.0, dtype=dtype)
    out = act(x)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 1e-2


@pytest.mark.parametrize("name", ["leaky_relu", "leaky_relu2"])
@pytest.mark.parametrize("dtype,atol", DTYPES)
def test_ungated_leaky_large_negative(name, dtype, atol):
    """Leaky variants don't saturate on negative; output is bounded but non-zero. Match ref."""
    act = UNGATED_ACTIVATIONS[name]
    ref = UNGATED_ACTIVATIONS_REFS[name]
    x = torch.full((2, 16, 64), -100.0, dtype=dtype)
    out = act(x)
    assert torch.isfinite(out).all()
    # leaky_relu(-100) = -1; leaky_relu2(-100) = 1.
    assert torch.allclose(out, ref(x), atol=max(atol, 1e-3))


# ---------------------------- Gated (GLU family) ----------------------------


@pytest.mark.parametrize("name", GATED_ACT_NAMES)
@pytest.mark.parametrize("dtype,atol", DTYPES)
def test_gated_matches_ref(name, dtype, atol):
    """Each gated activation matches act(gate) * up."""
    act = GATED_ACTIVATIONS[name]
    gate = torch.randn(2, 16, 64, dtype=dtype)
    up = torch.randn(2, 16, 64, dtype=dtype)
    out = act(gate, up)
    assert out.dtype == dtype
    # rtol scales tolerance with magnitude (output ~ act(gate)*up, can be O(10)).
    # powlu's compound op chain (sqrt + pow + sigmoid) accumulates more error,
    # so it needs a looser rtol in low precision than the simple gated forms.
    if dtype == torch.float32:
        rtol = 0.0
    elif dtype == torch.bfloat16:
        rtol = 6e-2 if name == "powlu" else 1e-2
    else:  # fp16
        rtol = 1e-2 if name == "powlu" else 2e-3
    assert torch.allclose(
        out, GATED_ACTIVATIONS_REFS[name](gate, up), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("name", SIMPLE_GATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_large_input_no_overflow_simple(name, dtype):
    """Large gate/up for simple gated activations: output finite, matches ref."""
    act = GATED_ACTIVATIONS[name]
    gate = torch.full((2, 16, 64), 100.0, dtype=dtype)
    up = torch.full((2, 16, 64), 100.0, dtype=dtype)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    # rtol scales with output magnitude — powlu(100)*100 ≈ 3.5e4, where 1 ULP
    # is ~32 (fp16) / ~256 (bf16), well over the atol tuned for outputs ~1e4.
    rtol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    assert torch.allclose(
        out, GATED_ACTIVATIONS_REFS[name](gate, up), atol=10.0, rtol=rtol
    )


@pytest.mark.parametrize("name", SQUARED_GATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_large_input_no_overflow_squared(name, dtype):
    """Squared gated act: act(20)² · 20 ≈ 8000 fits fp16 (max 65504)."""
    act = GATED_ACTIVATIONS[name]
    gate = torch.full((2, 16, 64), 20.0, dtype=dtype)
    up = torch.full((2, 16, 64), 20.0, dtype=dtype)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, GATED_ACTIVATIONS_REFS[name](gate, up), atol=100.0)


@pytest.mark.parametrize("name", GATED_ACT_NAMES)
def test_gated_zero_gate_zeros_output(name):
    """gate=0 → act(0)=0 → output=0 regardless of up."""
    act = GATED_ACTIVATIONS[name]
    gate = torch.zeros(2, 16, 64)
    up = torch.randn(2, 16, 64)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_powlu_negative_gate_saturates(dtype):
    """PowLU's negative branch is x² · sigmoid(x). For gate=-50, sigmoid(-50) ≈ 0
    so output ≈ 0 regardless of up. Important for stability under low-precision."""
    act = GATED_ACTIVATIONS["powlu"]
    gate = torch.full((2, 16, 64), -50.0, dtype=dtype)
    up = torch.full((2, 16, 64), 1.0, dtype=dtype)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 1e-3
