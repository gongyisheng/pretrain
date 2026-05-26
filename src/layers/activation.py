"""Activation functions used by FFN modules.

Two function families, both keyed by the same short activation name
(``"relu"``, ``"gelu"``, ``"silu"``):

- ``UNGATED_ACTIVATIONS``: unary ``x → act(x)`` — used by an ungated FFN as
  ``act(up_proj(x))``.
- ``GATED_ACTIVATIONS``: fused ``(gate, up) → act(gate) * up`` — used by a
  gated (GLU-family) FFN. Fused via ``@torch.compile``.

Literature names map onto our gated functions as:
    silu_glu = SwiGLU   (Shazeer 2020)
    gelu_glu = GeGLU
    relu_glu = ReGLU
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


# --- Gated (GLU) activations: (gate, up) → act(gate) * up, fused ---


@torch.compile
def relu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.relu(gate) * up


@torch.compile
def gelu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.gelu(gate) * up


@torch.compile
def silu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


@torch.compile
def leaky_relu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(gate, 0.01) * up


@torch.compile
def relu2_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.relu(gate) ** 2) * up


@torch.compile
def gelu2_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.gelu(gate) ** 2) * up


@torch.compile
def silu2_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.silu(gate) ** 2) * up


@torch.compile
def leaky_relu2_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.leaky_relu(gate, 0.01) ** 2) * up


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
}
