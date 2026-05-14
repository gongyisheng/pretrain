import torch
import torch.nn as nn

from src.layers.activation import GATED_ACTIVATIONS, UNGATED_ACTIVATIONS


class FFN(nn.Module):
    """Configurable feed-forward network.

    Two structural variants selected by `gated`:
      - gated=False: down_proj(activation(up_proj(x)))
      - gated=True (GLU family): down_proj(activation(gate_proj(x), up_proj(x)))

    `up_proj` and `down_proj` are always present; `gate_proj` is only added
    when `gated=True`. Names match the HF Qwen3 / Llama MLP convention.

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
            raise ValueError(f"Unknown activation: {activation!r}; expected one of {sorted(registry)}")
        self.gated = gated
        self.act_fn = registry[activation]
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=bias)
        if gated:
            self.gate_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            hidden = self.act_fn(self.gate_proj(x), self.up_proj(x))
        else:
            hidden = self.act_fn(self.up_proj(x))
        return self.dropout(self.down_proj(hidden))
