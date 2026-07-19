"""Activation functions used by FFN modules.

Two function families, keyed by the same short activation name:

- ``UNGATED_ACTIVATIONS``: unary ``x → act(x)`` — used by an ungated FFN as
  ``act(up_proj(x))``.
- ``GATED_ACTIVATIONS``: ``(gate, up) → act(gate) * up`` — used by a
  gated (GLU-family) FFN. These are plain elementwise ops; fusion comes from
  the whole-model ``torch.compile`` in the trainer, not a per-op decorator.

Squared variants follow **Rule A**: the unary activation is squared; the
gated form is ``squared_unary(gate) * up``. So ``silu2_glu`` is
``silu(gate)² * up`` (SwiGLU²), not ``(silu(gate) * up)²``.

Literature name mapping:

    relu_glu        = ReGLU       (Shazeer 2020)
    gelu_glu        = GeGLU
    silu_glu        = SwiGLU
    leaky_relu_glu  = LeakyReGLU
    relu2_glu       = ReGLU²      (Primer-style squared, gated)
    gelu2_glu       = GeGLU²
    silu2_glu       = SwiGLU²
    leaky_relu2_glu = LeakyReGLU²
    bilinear        = Bilinear GLU (Shazeer 2020) — gated-only
    bilinear2       = Squared Bilinear GLU — gated-only
    powlu           = PowLU       (arXiv:2605.25704, May 2026; m=3.0) — gated-only
"""

import torch
import torch.nn.functional as F


# --- Unary activations: x → act(x) ---


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, 0.01)


def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


def gelu2(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x) ** 2


def silu2(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x) ** 2


def leaky_relu2(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, 0.01) ** 2


# --- Gated (GLU) activations: (gate, up) → act(gate) * up ---


def relu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.relu(gate) * up


def gelu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.gelu(gate) * up


def silu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def leaky_relu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(gate, 0.01) * up


def relu2_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.relu(gate) ** 2) * up


def gelu2_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.gelu(gate) ** 2) * up


def silu2_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.silu(gate) ** 2) * up


def leaky_relu2_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.leaky_relu(gate, 0.01) ** 2) * up


def bilinear(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Bilinear GLU: gate * up. No unary activation. Shazeer 2020."""
    return gate * up


def bilinear2(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Squared Bilinear GLU: gate² * up."""
    return (gate**2) * up


def powlu(x: torch.Tensor) -> torch.Tensor:
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
    m = 3.0
    safe_x = torch.where(x > 0, x, torch.ones_like(x))
    exponent = m / (torch.sqrt(safe_x) + 1.0)
    pos = x * (safe_x**exponent) * torch.sigmoid(x)
    neg = x * x * torch.sigmoid(x)
    return torch.where(x > 0, pos, neg)


def powlu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return powlu(gate) * up


UNGATED_ACTIVATIONS = {
    "relu": relu,
    "gelu": gelu,
    "silu": silu,
    "leaky_relu": leaky_relu,
    "relu2": relu2,
    "gelu2": gelu2,
    "silu2": silu2,
    "leaky_relu2": leaky_relu2,
}
GATED_ACTIVATIONS = {
    "relu": relu_glu,
    "gelu": gelu_glu,
    "silu": silu_glu,
    "leaky_relu": leaky_relu_glu,
    "relu2": relu2_glu,
    "gelu2": gelu2_glu,
    "silu2": silu2_glu,
    "leaky_relu2": leaky_relu2_glu,
    "bilinear": bilinear,
    "bilinear2": bilinear2,
    "powlu": powlu_glu,
}
