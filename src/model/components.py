import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Norms ---

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# --- RoPE ---

class RotaryEmbedding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        inv_freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))  # (d_head//2,)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freqs)                                             # (max_seq_len, d_head//2)
        # HuggingFace-style split-half layout: cat along last dim for apply_rope
        self.register_buffer('cos', torch.cat([freqs.cos(), freqs.cos()], dim=-1))   # (max_seq_len, d_head)
        self.register_buffer('sin', torch.cat([freqs.sin(), freqs.sin()], dim=-1))   # (max_seq_len, d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, S, d_head)
        S = x.shape[2]
        cos = self.cos[:S].unsqueeze(0).unsqueeze(0)  # (1, 1, S, d_head)
        sin = self.sin[:S].unsqueeze(0).unsqueeze(0)  # (1, 1, S, d_head)
        return x * cos + _rotate_half(x) * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


# --- Attention ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, S, d_head)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, S, H)
        return self.resid_dropout(self.out_proj(out))


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rotary_emb = RotaryEmbedding(self.d_head, max_seq_len, rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)    # (B, n_heads, S, d_head)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2) # (B, n_kv_heads, S, d_head)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2) # (B, n_kv_heads, S, d_head)

        q = self.rotary_emb(q)
        k = self.rotary_emb(k)

        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, S, d_head)
        v = v.repeat_interleave(self.n_groups, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, S, H)
        return self.resid_dropout(self.out_proj(out))


# --- FFN ---

class GeluFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SwiGluFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = self.dropout(x)
        return x


# --- Transformer Block ---

class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = GeluFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
