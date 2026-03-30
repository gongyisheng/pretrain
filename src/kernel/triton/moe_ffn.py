import torch
import triton
import triton.language as tl
from src.kernel.triton.swiglu import triton_swiglu


def triton_moe_expert_ffn(
    padded_input: torch.Tensor,
    expert_gate_up: torch.Tensor,
    expert_down: torch.Tensor,
) -> torch.Tensor:
    """Batched expert FFN: gate_up bmm → SwiGLU → down bmm.

    Uses cuBLAS bmm for the matmuls and Triton SwiGLU for the activation.
    The SwiGLU kernel is element-wise, so it handles the 3D (E, C, I)
    shape by operating on the flattened view.

    Args:
        padded_input: (E, C, D) — padded token embeddings per expert
        expert_gate_up: (E, 2*I, D) — stacked gate+up projection weights
        expert_down: (E, D, I) — stacked down projection weights

    Returns:
        (E, C, D) — expert output
    """
    gate_up = torch.bmm(padded_input, expert_gate_up.mT)  # (E, C, 2*I)
    gate, up = gate_up.chunk(2, dim=-1)                     # each (E, C, I)
    hidden = triton_swiglu(gate.contiguous(), up.contiguous())  # (E, C, I)
    return torch.bmm(hidden, expert_down.mT)                # (E, C, D)
