import torch
import torch.nn.functional as F


@torch.compile
def torch_moe_expert_ffn(
    padded_input: torch.Tensor,
    expert_gate_up: torch.Tensor,
    expert_down: torch.Tensor,
) -> torch.Tensor:
    """Fused batched expert FFN: gate_up bmm → chunk → SwiGLU → down bmm.

    Compiling these ops together lets the compiler fuse the chunk + SwiGLU
    activation with the surrounding matmuls, reducing intermediate memory traffic.

    Args:
        padded_input: (E, C, D) — padded token embeddings per expert
        expert_gate_up: (E, 2*I, D) — stacked gate+up projection weights
        expert_down: (E, D, I) — stacked down projection weights

    Returns:
        (E, C, D) — expert output
    """
    gate_up = torch.bmm(padded_input, expert_gate_up.mT)  # (E, C, 2*I)
    gate, up = gate_up.chunk(2, dim=-1)                     # each (E, C, I)
    hidden = F.silu(gate) * up                               # (E, C, I)
    return torch.bmm(hidden, expert_down.mT)                # (E, C, D)
