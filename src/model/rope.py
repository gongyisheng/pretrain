import torch
import torch.nn as nn


@torch.compile
def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fused: rotate-half → scale by cos/sin → add."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class RoPE(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.theta = theta
        self.max_seq_len = max_seq_len
        self._build_buffers()

    def _build_buffers(self):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_head, 2) / self.d_head))
        positions = torch.arange(self.max_seq_len)
        angles = positions[:, None] * freqs[None, :]          # (max_seq_len, d_head//2)
        angles = torch.cat([angles, angles], dim=-1)           # (max_seq_len, d_head)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings.

        Args:
            x: shape (B, n_heads, S, d_head)
            position_ids: shape (B, S) — per-token position used to gather
                cos/sin tables, supporting per-document position resets.
        """
        # TODO: the fancy index performance is bad, need to fuse into the rope op
        cos = self.cos[position_ids][:, None, :, :].to(x.dtype)  # (B, 1, S, d_head)
        sin = self.sin[position_ids][:, None, :, :].to(x.dtype)  # (B, 1, S, d_head)
        return _apply_rope(x, cos, sin)
