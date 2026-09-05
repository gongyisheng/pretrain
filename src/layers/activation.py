from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F


# --- Unary activations: x → act(x) ---


def relu(x: torch.Tensor, act_limit: dict | None = None) -> torch.Tensor:
    if act_limit and "up" in act_limit:
        x = x.clamp(**act_limit["up"])
    return F.relu(x)


def gelu(x: torch.Tensor, act_limit: dict | None = None) -> torch.Tensor:
    if act_limit and "up" in act_limit:
        x = x.clamp(**act_limit["up"])
    return F.gelu(x)


def silu(
    x: torch.Tensor, alpha: float = 1.0, act_limit: dict | None = None
) -> torch.Tensor:
    """x * sigmoid(alpha * x). alpha scales only the sigmoid argument, so
    alpha=1.0 is exactly F.silu (gpt-oss uses alpha=1.702)."""
    if act_limit and "up" in act_limit:
        x = x.clamp(**act_limit["up"])
    if alpha == 1.0:
        return F.silu(x)
    return x * torch.sigmoid(alpha * x)


def leaky_relu(
    x: torch.Tensor, negative_slope: float = 0.01, act_limit: dict | None = None
) -> torch.Tensor:
    if act_limit and "up" in act_limit:
        x = x.clamp(**act_limit["up"])
    return F.leaky_relu(x, negative_slope)


def relu2(x: torch.Tensor, act_limit: dict | None = None) -> torch.Tensor:
    if act_limit and "up" in act_limit:
        x = x.clamp(**act_limit["up"])
    return F.relu(x) ** 2


def gelu2(x: torch.Tensor, act_limit: dict | None = None) -> torch.Tensor:
    if act_limit and "up" in act_limit:
        x = x.clamp(**act_limit["up"])
    return F.gelu(x) ** 2


def silu2(
    x: torch.Tensor, alpha: float = 1.0, act_limit: dict | None = None
) -> torch.Tensor:
    if act_limit and "up" in act_limit:
        x = x.clamp(**act_limit["up"])
    if alpha == 1.0:
        return F.silu(x) ** 2
    return (x * torch.sigmoid(alpha * x)) ** 2


def leaky_relu2(
    x: torch.Tensor, negative_slope: float = 0.01, act_limit: dict | None = None
) -> torch.Tensor:
    if act_limit and "up" in act_limit:
        x = x.clamp(**act_limit["up"])
    return F.leaky_relu(x, negative_slope) ** 2


# --- Gated (GLU) activations: (gate, up) → act(gate) * (up + up_shift) ---


def reglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    return F.relu(gate) * up


def geglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    return F.gelu(gate) * up


def swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float = 1.0,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    """
    SwiGLU: SiLU's gated form.
    gpt-oss's variant is alpha=1.702, up_shift=1.0, act_limit=±7.
    """
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    if alpha == 1.0:
        return F.silu(gate) * up
    return gate * torch.sigmoid(alpha * gate) * up


def leaky_reglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    negative_slope: float = 0.01,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    return F.leaky_relu(gate, negative_slope) * up


def reglu2(
    gate: torch.Tensor,
    up: torch.Tensor,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    return (F.relu(gate) ** 2) * up


def geglu2(
    gate: torch.Tensor,
    up: torch.Tensor,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    return (F.gelu(gate) ** 2) * up


def swiglu2(
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float = 1.0,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    if alpha == 1.0:
        return (F.silu(gate) ** 2) * up
    return ((gate * torch.sigmoid(alpha * gate)) ** 2) * up


def leaky_reglu2(
    gate: torch.Tensor,
    up: torch.Tensor,
    negative_slope: float = 0.01,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    return (F.leaky_relu(gate, negative_slope) ** 2) * up


def bilinear(
    gate: torch.Tensor,
    up: torch.Tensor,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    """Bilinear GLU: gate * up. No unary activation. Shazeer 2020."""
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    return gate * up


def bilinear2(
    gate: torch.Tensor,
    up: torch.Tensor,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    """Squared Bilinear GLU: gate² * up."""
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    return (gate**2) * up


def powlu(
    gate: torch.Tensor,
    up: torch.Tensor,
    m: float = 3.0,
    up_shift: float = 0.0,
    act_limit: dict | None = None,
) -> torch.Tensor:
    """PowLU helper: arXiv:2605.25704 (May 2026), m=3.0.

    Piecewise:
        x > 0:  x · x^(m/(sqrt(x)+1)) · sigmoid(x)
        x <= 0: x² · sigmoid(x)

    The √x denominator makes large positive growth sub-quadratic, reducing
    activation outliers vs. SwiGLU's near-x² growth.

    `safe_x = where(x > 0, x, 1.0)` serves a dual purpose so the pos branch
    is NaN-free everywhere (required for torch.compile tracing and autograd
    through the unselected branch):
      - sqrt argument: avoids sqrt(negative) on the discarded path.
      - pow base: avoids negative ** non_integer on the discarded path.

    The outer `torch.where(x > 0, pos, neg)` selects the correct branch;
    when x > 0, safe_x == x, so the forward value is unchanged.
    """
    if act_limit:
        if "gate" in act_limit:
            gate = gate.clamp(**act_limit["gate"])
        if "up" in act_limit:
            up = up.clamp(**act_limit["up"])
    if up_shift != 0.0:
        up = up + up_shift
    safe_gate = torch.where(gate > 0, gate, torch.ones_like(gate))
    exponent = m / (torch.sqrt(safe_gate) + 1.0)
    pos = gate * (safe_gate**exponent) * torch.sigmoid(gate)
    neg = gate * gate * torch.sigmoid(gate)
    return torch.where(gate > 0, pos, neg) * up


class Activation(NamedTuple):
    """Registry entry: the function, plus the arity its name already implies."""

    fn: Callable
    gated: bool


# The name alone says gated or unary -- "swiglu" is SiLU's gated form, "silu" the
# unary one -- so no separate `gated` config key is needed.
ACT_REGISTRY = {
    "relu": Activation(relu, False),
    "gelu": Activation(gelu, False),
    "silu": Activation(silu, False),
    "leaky_relu": Activation(leaky_relu, False),
    "relu2": Activation(relu2, False),
    "gelu2": Activation(gelu2, False),
    "silu2": Activation(silu2, False),
    "leaky_relu2": Activation(leaky_relu2, False),
    "reglu": Activation(reglu, True),
    "geglu": Activation(geglu, True),
    "swiglu": Activation(swiglu, True),
    "leaky_reglu": Activation(leaky_reglu, True),
    "reglu2": Activation(reglu2, True),
    "geglu2": Activation(geglu2, True),
    "swiglu2": Activation(swiglu2, True),
    "leaky_reglu2": Activation(leaky_reglu2, True),
    "bilinear": Activation(bilinear, True),
    "bilinear2": Activation(bilinear2, True),
    "powlu": Activation(powlu, True),
}
