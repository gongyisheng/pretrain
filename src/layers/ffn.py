import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.compile
def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused: silu(gate) * up in a single pass."""
    return F.silu(gate) * up


class GeluFFN(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int, dropout_ffn: float = 0.0, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SwiGluFFN(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int, dropout_ffn: float = 0.0, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.down_proj(_swiglu(gate, up))
        x = self.dropout(x)
        return x
