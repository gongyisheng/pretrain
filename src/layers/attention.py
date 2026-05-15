from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.norm import RMSNorm

if TYPE_CHECKING:
    from src.layers.pos_emb import RoPE


@torch.compile
def _flash_attn(q, k, v, attn_mask=None, is_causal=False, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=sm_scale
    )


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_attn: float = 0.0, qk_norm: bool = False, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout_attn)

        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor, rope: "RoPE" = None, position_ids: torch.Tensor = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, S, H = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, S, d_head)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q.reshape(-1, S, self.d_head)).view(B, self.n_heads, S, self.d_head)
            k = self.k_norm(k.reshape(-1, S, self.d_head)).view(B, self.n_heads, S, self.d_head)

        if rope is not None:
            assert position_ids is not None, "position_ids cannot be None when using RoPE"
            q = rope(q, position_ids=position_ids)
            k = rope(k, position_ids=position_ids)

        is_causal = attn_mask is None
        out = _flash_attn(q, k, v, is_causal=is_causal, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, S, H)
        return self.attn_dropout(self.o_proj(out))


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout_attn: float = 0.0,
        qk_norm: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout_attn)

        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor, rope: "RoPE" = None, position_ids: torch.Tensor = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, S, H = x.shape

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)    # (B, n_heads, S, d_head)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2) # (B, n_kv_heads, S, d_head)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2) # (B, n_kv_heads, S, d_head)

        if self.qk_norm:
            q = self.q_norm(q.reshape(-1, S, self.d_head)).view(B, self.n_heads, S, self.d_head)
            k = self.k_norm(k.reshape(-1, S, self.d_head)).view(B, self.n_kv_heads, S, self.d_head)

        if rope is not None:
            assert position_ids is not None, "position_ids cannot be None when using RoPE"
            q = rope(q, position_ids)
            k = rope(k, position_ids)

        # Expand KV heads for GQA (expand+reshape avoids memory allocation vs repeat_interleave)
        k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_groups, S, self.d_head).reshape(B, self.n_heads, S, self.d_head)
        v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_groups, S, self.d_head).reshape(B, self.n_heads, S, self.d_head)

        is_causal = attn_mask is None
        out = _flash_attn(q, k, v, is_causal=is_causal, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, S, H)
        return self.attn_dropout(self.o_proj(out))
