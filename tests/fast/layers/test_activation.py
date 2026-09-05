"""Numerical parity tests for activation functions vs eager mathematical refs."""

import pytest
import torch

from src.layers.activation import ACT_REGISTRY
from tests.fast.layers._refs import REFS, SIMPLE_DTYPES


DTYPES = SIMPLE_DTYPES
# One registry; the entry carries the arity, so the name alone picks gate-vs-unary.
ACT_NAMES = [n for n, e in ACT_REGISTRY.items() if not e.gated]
GATED_ACT_NAMES = [n for n, e in ACT_REGISTRY.items() if e.gated]
# Sub-groups for tests that need different input magnitudes or behavior assumptions.
SIMPLE_UNGATED = ["relu", "gelu", "silu", "leaky_relu"]
SQUARED_UNGATED = ["relu2", "gelu2", "silu2", "leaky_relu2"]
# Ungated names whose output saturates to ~0 for large negative x.
# leaky variants are excluded — they preserve a small slope on the negative side.
SATURATING_UNGATED = [
    "relu",
    "gelu",
    "silu",
    "relu2",
    "gelu2",
    "silu2",
]
SIMPLE_GATED = ["reglu", "geglu", "swiglu", "leaky_reglu", "bilinear", "powlu"]
SQUARED_GATED = ["reglu2", "geglu2", "swiglu2", "leaky_reglu2", "bilinear2"]


# ---------------------------- Ungated ----------------------------


@pytest.mark.parametrize("name", ACT_NAMES)
@pytest.mark.parametrize("dtype,atol", DTYPES)
def test_ungated_matches_ref(name, dtype, atol):
    """Each ungated activation matches its eager mathematical definition."""
    act = ACT_REGISTRY[name].fn
    ref = REFS[name]
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
    act = ACT_REGISTRY[name].fn
    x = torch.zeros(8, 64)
    out = act(x)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


@pytest.mark.parametrize("name", SIMPLE_UNGATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ungated_large_positive_no_overflow_simple(name, dtype):
    """Large positive x: simple unary acts pass through ~unchanged; output must be finite."""
    act = ACT_REGISTRY[name].fn
    ref = REFS[name]
    x = torch.full((2, 16, 64), 1000.0, dtype=dtype)
    out = act(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref(x), atol=1.0)


@pytest.mark.parametrize("name", SQUARED_UNGATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ungated_large_positive_no_overflow_squared(name, dtype):
    """Large positive x for squared variants: 200² = 4e4 fits fp16 (max 65504)."""
    act = ACT_REGISTRY[name].fn
    ref = REFS[name]
    x = torch.full((2, 16, 64), 200.0, dtype=dtype)
    out = act(x)
    assert torch.isfinite(out).all()
    # Squared output is ~4e4; atol scales with magnitude.
    assert torch.allclose(out, ref(x), atol=100.0)


@pytest.mark.parametrize("name", SATURATING_UNGATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ungated_large_negative_saturates(name, dtype):
    """Large negative x: relu/gelu/silu all → ~0; sigmoid/exp must not underflow to nan."""
    act = ACT_REGISTRY[name].fn
    x = torch.full((2, 16, 64), -1000.0, dtype=dtype)
    out = act(x)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 1e-2


@pytest.mark.parametrize("name", ["leaky_relu", "leaky_relu2"])
@pytest.mark.parametrize("dtype,atol", DTYPES)
def test_ungated_leaky_large_negative(name, dtype, atol):
    """Leaky variants don't saturate on negative; output is bounded but non-zero. Match ref."""
    act = ACT_REGISTRY[name].fn
    ref = REFS[name]
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
    act = ACT_REGISTRY[name].fn
    gate = torch.randn(2, 16, 64, dtype=dtype)
    up = torch.randn(2, 16, 64, dtype=dtype)
    out = act(gate, up)
    assert out.dtype == dtype
    # rtol scales tolerance with magnitude (output ~ act(gate)*up, can be O(10)).
    # powlu's compound op chain (sqrt + pow + sigmoid) accumulates more error,
    # and squared variants double the base activation's relative error, so both
    # need a looser rtol in low precision than the simple gated forms.
    squared = name.endswith("2")
    if dtype == torch.float32:
        rtol = 0.0
    elif dtype == torch.bfloat16:
        rtol = 6e-2 if name == "powlu" else (2e-2 if squared else 1e-2)
    else:  # fp16
        rtol = 1e-2 if name == "powlu" else 2e-3
    assert torch.allclose(out, REFS[name](gate, up), atol=atol, rtol=rtol)


@pytest.mark.parametrize("name", SIMPLE_GATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_large_input_no_overflow_simple(name, dtype):
    """Large gate/up for simple gated activations: output finite, matches ref."""
    act = ACT_REGISTRY[name].fn
    gate = torch.full((2, 16, 64), 100.0, dtype=dtype)
    up = torch.full((2, 16, 64), 100.0, dtype=dtype)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    # rtol scales with output magnitude — powlu(100)*100 ≈ 3.5e4, where 1 ULP
    # is ~32 (fp16) / ~256 (bf16), well over the atol tuned for outputs ~1e4.
    rtol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    assert torch.allclose(out, REFS[name](gate, up), atol=10.0, rtol=rtol)


@pytest.mark.parametrize("name", SQUARED_GATED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_large_input_no_overflow_squared(name, dtype):
    """Squared gated act: act(20)² · 20 ≈ 8000 fits fp16 (max 65504)."""
    act = ACT_REGISTRY[name].fn
    gate = torch.full((2, 16, 64), 20.0, dtype=dtype)
    up = torch.full((2, 16, 64), 20.0, dtype=dtype)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, REFS[name](gate, up), atol=100.0)


@pytest.mark.parametrize("name", GATED_ACT_NAMES)
def test_gated_zero_gate_zeros_output(name):
    """gate=0 → act(0)=0 → output=0 regardless of up."""
    act = ACT_REGISTRY[name].fn
    gate = torch.zeros(2, 16, 64)
    up = torch.randn(2, 16, 64)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_powlu_negative_gate_saturates(dtype):
    """PowLU's negative branch is x² · sigmoid(x). For gate=-50, sigmoid(-50) ≈ 0
    so output ≈ 0 regardless of up. Important for stability under low-precision."""
    act = ACT_REGISTRY["powlu"].fn
    gate = torch.full((2, 16, 64), -50.0, dtype=dtype)
    up = torch.full((2, 16, 64), 1.0, dtype=dtype)
    out = act(gate, up)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 1e-3


# ---------------------------- activation_kwargs axes ----------------------------

# Case lists are independent axes: any activation crosses any limit/shift/alpha.
UNGATED_LIMITS = [
    {"up": {"max": 1.0}},
    {"up": {"min": -1.0}},
    {"up": {"min": -1.0, "max": 1.0}},
]
GATED_LIMITS = [
    {"gate": {"max": 1.0}},
    {"up": {"min": -1.0, "max": 1.0}},
    {"gate": {"max": 7.0}, "up": {"min": -7.0, "max": 7.0}},
]
UP_SHIFTS = [1.0, -0.5]
ALPHAS = [1.702, 0.5]
ALPHA_ACTS = ["silu", "silu2", "swiglu", "swiglu2"]
# Already floored at 0 on the negative side, so a min-only bound cannot bite.
FLOORED_UNGATED = ["relu", "relu2"]
SLOPE_ACTS = ["leaky_relu", "leaky_relu2"]


def _spread(*shape, dtype=torch.float32, seed=0):
    """Inputs wide enough that every limit in the case lists actually bites."""
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype) * 5.0


@pytest.mark.parametrize("name", ACT_NAMES)
@pytest.mark.parametrize("act_limit", UNGATED_LIMITS)
def test_ungated_act_limit(name, act_limit):
    """act_limit clamps the input before the activation, and actually bites."""
    if name in FLOORED_UNGATED and set(act_limit["up"]) == {"min"}:
        pytest.skip(f"{name} floors negatives at 0; a min-only bound is a no-op")
    act = ACT_REGISTRY[name].fn
    x = _spread(2, 16, 64)
    # same ops on both sides -> bit-identical, so exact comparison
    assert torch.equal(act(x, act_limit=act_limit), act(x.clamp(**act_limit["up"])))
    assert not torch.equal(act(x, act_limit=act_limit), act(x))
    assert torch.equal(act(x, act_limit=None), act(x))


@pytest.mark.parametrize("name", GATED_ACT_NAMES)
@pytest.mark.parametrize("act_limit", GATED_LIMITS)
def test_gated_act_limit(name, act_limit):
    """Gated act_limit clamps each side independently, pre-activation."""
    act = ACT_REGISTRY[name].fn
    gate, up = _spread(2, 16, 64, seed=1), _spread(2, 16, 64, seed=2)
    g = gate.clamp(**act_limit["gate"]) if "gate" in act_limit else gate
    u = up.clamp(**act_limit["up"]) if "up" in act_limit else up
    assert torch.equal(act(gate, up, act_limit=act_limit), act(g, u))
    assert not torch.equal(act(gate, up, act_limit=act_limit), act(gate, up))


@pytest.mark.parametrize("name", GATED_ACT_NAMES)
@pytest.mark.parametrize("up_shift", UP_SHIFTS)
def test_gated_up_shift(name, up_shift):
    """up_shift offsets the up branch; the default 0.0 stays bit-identical."""
    act = ACT_REGISTRY[name].fn
    gate, up = _spread(2, 16, 64, seed=1), _spread(2, 16, 64, seed=2)
    assert torch.equal(act(gate, up, up_shift=up_shift), act(gate, up + up_shift))
    assert torch.equal(act(gate, up, up_shift=0.0), act(gate, up))


@pytest.mark.parametrize("name", ALPHA_ACTS)
@pytest.mark.parametrize("alpha", ALPHAS)
def test_silu_alpha(name, alpha):
    """alpha scales only the sigmoid argument; alpha=1.0 stays bit-identical."""
    x = _spread(2, 16, 64)
    act, gated = ACT_REGISTRY[name]
    args = (x, x) if gated else (x,)
    assert torch.equal(act(*args, alpha=1.0), act(*args))
    assert not torch.equal(act(*args, alpha=alpha), act(*args))


@pytest.mark.parametrize("alpha", [1.0, *ALPHAS])
@pytest.mark.parametrize("dtype,atol", DTYPES)
def test_silu_alpha_precision(alpha, dtype, atol):
    """silu(x, alpha) == x * sigmoid(alpha*x), including the alpha=1 F.silu path."""
    x = torch.randn(2, 16, 64, dtype=dtype)
    xf = x.float()
    ref = (xf * torch.sigmoid(alpha * xf)).to(dtype)
    out = ACT_REGISTRY["silu"].fn(x, alpha=alpha)
    assert out.dtype == dtype
    assert torch.allclose(out, ref, atol=atol, rtol=0.0)


@pytest.mark.parametrize("name", SLOPE_ACTS)
@pytest.mark.parametrize("negative_slope", [0.2, 0.0])
def test_leaky_relu_negative_slope(name, negative_slope):
    """negative_slope is configurable; the 0.01 default stays bit-identical."""
    x = _spread(2, 16, 64)
    act = ACT_REGISTRY[name].fn
    squared = 2 if name.endswith("2") else 1
    ref = torch.where(x > 0, x, negative_slope * x) ** squared
    assert torch.allclose(act(x, negative_slope=negative_slope), ref, atol=1e-6, rtol=0)
    assert torch.equal(act(x, negative_slope=0.01), act(x))


@pytest.mark.parametrize("m", [2.0, 4.0])
def test_powlu_m(m):
    """powlu's exponent m is configurable; the 3.0 default stays bit-identical."""
    act = ACT_REGISTRY["powlu"].fn
    gate, up = _spread(2, 16, 64, seed=1), _spread(2, 16, 64, seed=2)
    assert torch.equal(act(gate, up, m=3.0), act(gate, up))
    assert not torch.equal(act(gate, up, m=m), act(gate, up))
