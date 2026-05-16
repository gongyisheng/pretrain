import torch
import torch.nn as nn

from src.layers.activation import GATED_ACTIVATIONS, UNGATED_ACTIVATIONS


@torch.compile
def gated_ffn(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    act_fn,
    b_gate_up: torch.Tensor = None,
    b_down: torch.Tensor = None,
) -> torch.Tensor:
    """Fused gated FFN: gate_up matmul → chunk → act(gate, up) → down matmul.

    Shapes are unified via `@` (matmul broadcasts arbitrary leading dims):
      - Dense (used by FFN):     x (..., D); w_gate_up (2*I, D); w_down (D, I).
      - Batched (used by MoE):   x (E, C, D); w_gate_up (E, 2*I, D); w_down (E, D, I).
    Bias is 1D for the dense form and 2D for the batched form (auto-broadcasts).
    """
    gate_up = x @ w_gate_up.mT
    if b_gate_up is not None:
        gate_up = gate_up + (
            b_gate_up if b_gate_up.ndim == 1 else b_gate_up.unsqueeze(-2)
        )
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = act_fn(gate, up)
    out = hidden @ w_down.mT
    if b_down is not None:
        out = out + (b_down if b_down.ndim == 1 else b_down.unsqueeze(-2))
    return out


@torch.compile
def ungated_ffn(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    act_fn,
    b_up: torch.Tensor = None,
    b_down: torch.Tensor = None,
) -> torch.Tensor:
    """Fused ungated FFN: up matmul → act(up) → down matmul.

    Shapes (parallel to gated_ffn):
      - Dense:   x (..., D); w_up (I, D); w_down (D, I).
      - Batched: x (E, C, D); w_up (E, I, D); w_down (E, D, I).
    """
    up = x @ w_up.mT
    if b_up is not None:
        up = up + (b_up if b_up.ndim == 1 else b_up.unsqueeze(-2))
    hidden = act_fn(up)
    out = hidden @ w_down.mT
    if b_down is not None:
        out = out + (b_down if b_down.ndim == 1 else b_down.unsqueeze(-2))
    return out


class FFN(nn.Module):
    """Configurable feed-forward network.

    Two structural variants selected by `gated`:
      - gated=False: down_proj(activation(up_proj(x)))
      - gated=True (GLU family): down_proj(activation(gate, up)) where
        (gate, up) = chunk(gate_up_proj(x), 2, dim=-1)

    Always present: `down_proj`. Then:
      - gated=False: `up_proj`
      - gated=True:  `gate_up_proj` (fused gate + up; one matmul instead of two)

    `activation` is one of ``"relu"`` / ``"gelu"`` / ``"silu"``. Combinations:
        gated=True + activation="silu" → SwiGLU
        gated=True + activation="gelu" → GeGLU
        gated=True + activation="relu" → ReGLU

    Bias defaults to False (modern LLM convention); pass `bias=True` for the
    GPT-2 / classic Transformer convention.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        *,
        activation: str = "gelu",
        gated: bool = False,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        registry = GATED_ACTIVATIONS if gated else UNGATED_ACTIVATIONS
        if activation not in registry:
            raise ValueError(
                f"Unknown activation: {activation!r}; expected one of {sorted(registry)}"
            )
        self.gated = gated
        self.act_fn = registry[activation]
        if gated:
            self.gate_up_proj = nn.Linear(d_model, 2 * intermediate_size, bias=bias)
        else:
            self.up_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            out = gated_ffn(
                x,
                self.gate_up_proj.weight,
                self.down_proj.weight,
                self.act_fn,
                self.gate_up_proj.bias,
                self.down_proj.bias,
            )
        else:
            out = ungated_ffn(
                x,
                self.up_proj.weight,
                self.down_proj.weight,
                self.act_fn,
                self.up_proj.bias,
                self.down_proj.bias,
            )
        return self.dropout(out)
