from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from src.layers.norm import RMSNorm

if TYPE_CHECKING:
    from src.layers.pos_emb import RoPE


@torch.compile
def _sdpa(q, k, v, attn_mask=None, is_causal=False, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=sm_scale
    )


_flex_attn = torch.compile(flex_attention)


def _call_sdpa(q, k, v, attn_mask):
    """SDPA path: ``attn_mask`` is either ``None`` (→ ``is_causal=True``) or a
    dense additive tensor."""
    is_causal = attn_mask is None
    return _sdpa(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


def _call_flex(q, k, v, attn_mask):
    """FlexAttention path: ``attn_mask`` must be a ``BlockMask``."""
    return _flex_attn(q, k, v, block_mask=attn_mask)


_ATTN_IMPL = {
    "sdpa": _call_sdpa,
    "flex_attention": _call_flex,
}


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        bias: bool = False,
        attn_implementation: str = "flex_attention",
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

        self._attn_fn = _ATTN_IMPL[attn_implementation]

    @classmethod
    def compute_flops(
        cls, d_model, max_seq_len, *, n_heads, bias=False, qk_norm=False, **_
    ):
        head_dim = d_model // n_heads
        n_kv = n_heads
        qkv = 2 * d_model * (n_heads + 2 * n_kv) * head_dim
        if bias:
            qkv += (n_heads + 2 * n_kv) * head_dim
        o = 2 * d_model * d_model + (d_model if bias else 0)
        attn_matmul = 4 * n_heads * head_dim * max_seq_len
        qk = (3 * (n_heads + n_kv) * head_dim) if qk_norm else 0
        return qkv + o + attn_matmul + qk

    @classmethod
    def compute_parameters(
        cls, d_model, *, n_heads, bias=False, qk_norm=False, **_
    ) -> int:
        head_dim = d_model // n_heads
        n_kv = n_heads
        qkv = d_model * (n_heads + 2 * n_kv) * head_dim
        if bias:
            qkv += (n_heads + 2 * n_kv) * head_dim
        o = d_model * d_model + (d_model if bias else 0)
        qk = (2 * head_dim) if qk_norm else 0  # q_norm + k_norm RMSNorm(head_dim)
        return qkv + o + qk

    def forward(
        self,
        x: torch.Tensor,
        rope: "RoPE" = None,
        position_ids: torch.Tensor = None,
        attn_mask=None,
    ) -> torch.Tensor:
        B, S, H = x.shape
        q = (
            self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        )  # (B, n_heads, S, d_head)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q.reshape(-1, S, self.d_head)).view(
                B, self.n_heads, S, self.d_head
            )
            k = self.k_norm(k.reshape(-1, S, self.d_head)).view(
                B, self.n_heads, S, self.d_head
            )

        if rope is not None:
            assert position_ids is not None, (
                "position_ids cannot be None when using RoPE"
            )
            q = rope(q, position_ids=position_ids)
            k = rope(k, position_ids=position_ids)

        out = self._attn_fn(q, k, v, attn_mask)
        out = out.transpose(1, 2).reshape(B, S, H)
        return self.attn_dropout(self.o_proj(out))


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        bias: bool = False,
        attn_implementation: str = "flex_attention",
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
        self.attn_dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

        self._attn_fn = _ATTN_IMPL[attn_implementation]

    @classmethod
    def compute_flops(
        cls,
        d_model,
        max_seq_len,
        *,
        n_heads,
        n_kv_heads=None,
        bias=False,
        qk_norm=False,
        **_,
    ):
        n_kv = n_kv_heads or n_heads
        head_dim = d_model // n_heads
        qkv = 2 * d_model * (n_heads + 2 * n_kv) * head_dim
        if bias:
            qkv += (n_heads + 2 * n_kv) * head_dim
        o = 2 * d_model * d_model + (d_model if bias else 0)
        attn_matmul = 4 * n_heads * head_dim * max_seq_len
        qk = (3 * (n_heads + n_kv) * head_dim) if qk_norm else 0
        return qkv + o + attn_matmul + qk

    @classmethod
    def compute_parameters(
        cls, d_model, *, n_heads, n_kv_heads=None, bias=False, qk_norm=False, **_
    ) -> int:
        n_kv = n_kv_heads or n_heads
        head_dim = d_model // n_heads
        qkv = d_model * (n_heads + 2 * n_kv) * head_dim
        if bias:
            qkv += (n_heads + 2 * n_kv) * head_dim
        o = d_model * d_model + (d_model if bias else 0)
        qk = (2 * head_dim) if qk_norm else 0  # q_norm + k_norm RMSNorm(head_dim)
        return qkv + o + qk

    def forward(
        self,
        x: torch.Tensor,
        rope: "RoPE" = None,
        position_ids: torch.Tensor = None,
        attn_mask=None,
    ) -> torch.Tensor:
        B, S, H = x.shape

        q = (
            self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        )  # (B, n_heads, S, d_head)
        k = (
            self.k_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        )  # (B, n_kv_heads, S, d_head)
        v = (
            self.v_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        )  # (B, n_kv_heads, S, d_head)

        if self.qk_norm:
            q = self.q_norm(q.reshape(-1, S, self.d_head)).view(
                B, self.n_heads, S, self.d_head
            )
            k = self.k_norm(k.reshape(-1, S, self.d_head)).view(
                B, self.n_kv_heads, S, self.d_head
            )

        if rope is not None:
            assert position_ids is not None, (
                "position_ids cannot be None when using RoPE"
            )
            q = rope(q, position_ids)
            k = rope(k, position_ids)

        # Expand KV heads for GQA (expand+reshape avoids memory allocation vs repeat_interleave)
        k = (
            k[:, :, None, :, :]
            .expand(B, self.n_kv_heads, self.n_groups, S, self.d_head)
            .reshape(B, self.n_heads, S, self.d_head)
        )
        v = (
            v[:, :, None, :, :]
            .expand(B, self.n_kv_heads, self.n_groups, S, self.d_head)
            .reshape(B, self.n_heads, S, self.d_head)
        )

        out = self._attn_fn(q, k, v, attn_mask)
        out = out.transpose(1, 2).reshape(B, S, H)
        return self.attn_dropout(self.o_proj(out))


class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2/V3).

    KV is compressed to a single ``kv_lora_rank`` latent (the only thing a KV
    cache would store) and up-projected to per-head K/V. RoPE can't fold through
    the latent up-projection, so position is carried by a *decoupled* rope part:
    each head's Q/K is ``[nope ; rope]`` where the K rope part (``qk_rope_head_dim``)
    is shared across heads and broadcast. Queries optionally share a ``q_lora_rank``
    latent too (params/compute saving, not a cache saving).

    Head dims are explicit; ``ModelConfig.__post_init__`` fills their defaults.
    ``q_lora_rank=0`` disables query compression (a plain ``q_proj`` is used).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: int = 0,
        dropout: float = 0.0,
        bias: bool = False,
        attn_implementation: str = "flex_attention",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        if q_lora_rank > 0:
            self.q_a_proj = nn.Linear(d_model, q_lora_rank, bias=bias)
            self.q_a_norm = RMSNorm(q_lora_rank)
            self.q_b_proj = nn.Linear(q_lora_rank, n_heads * qk_head_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(d_model, n_heads * qk_head_dim, bias=bias)

        # kv_a_proj emits the latent plus the shared (head-agnostic) rope key.
        self.kv_a_proj = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim, bias=bias)
        self.kv_a_norm = RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=bias
        )
        self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self._attn_fn = _ATTN_IMPL[attn_implementation]

    @classmethod
    def compute_flops(
        cls,
        d_model,
        max_seq_len,
        *,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        kv_lora_rank,
        q_lora_rank=0,
        bias=False,
        **_,
    ):
        qk_head = qk_nope_head_dim + qk_rope_head_dim
        b = lambda out: out if bias else 0  # noqa: E731
        if q_lora_rank > 0:
            q = (
                2 * d_model * q_lora_rank
                + b(q_lora_rank)
                + 3 * q_lora_rank  # q_a_norm
                + 2 * q_lora_rank * (n_heads * qk_head)
                + b(n_heads * qk_head)
            )
        else:
            q = 2 * d_model * (n_heads * qk_head) + b(n_heads * qk_head)
        kv_a = (
            2 * d_model * (kv_lora_rank + qk_rope_head_dim)
            + b(kv_lora_rank + qk_rope_head_dim)
            + 3 * kv_lora_rank  # kv_a_norm
        )
        kv_b = 2 * kv_lora_rank * (n_heads * (qk_nope_head_dim + v_head_dim)) + b(
            n_heads * (qk_nope_head_dim + v_head_dim)
        )
        o = 2 * (n_heads * v_head_dim) * d_model + b(d_model)
        attn_matmul = (
            2 * n_heads * qk_head * max_seq_len + 2 * n_heads * v_head_dim * max_seq_len
        )
        return q + kv_a + kv_b + o + attn_matmul

    @classmethod
    def compute_parameters(
        cls,
        d_model,
        *,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        kv_lora_rank,
        q_lora_rank=0,
        bias=False,
        **_,
    ) -> int:
        qk_head = qk_nope_head_dim + qk_rope_head_dim
        b = lambda out: out if bias else 0  # noqa: E731
        if q_lora_rank > 0:
            q = (
                d_model * q_lora_rank
                + b(q_lora_rank)
                + q_lora_rank  # q_a_norm weight
                + q_lora_rank * (n_heads * qk_head)
                + b(n_heads * qk_head)
            )
        else:
            q = d_model * (n_heads * qk_head) + b(n_heads * qk_head)
        kv_a = (
            d_model * (kv_lora_rank + qk_rope_head_dim)
            + b(kv_lora_rank + qk_rope_head_dim)
            + kv_lora_rank  # kv_a_norm weight
        )
        kv_b = kv_lora_rank * (n_heads * (qk_nope_head_dim + v_head_dim)) + b(
            n_heads * (qk_nope_head_dim + v_head_dim)
        )
        o = (n_heads * v_head_dim) * d_model + b(d_model)
        return q + kv_a + kv_b + o

    def forward(
        self,
        x: torch.Tensor,
        rope: "RoPE" = None,
        position_ids: torch.Tensor = None,
        attn_mask=None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H, nope, rdim, vdim = (
            self.n_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )

        if self.q_lora_rank > 0:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.view(B, S, H, nope + rdim).transpose(1, 2)  # (B, H, S, nope+rdim)
        q_nope, q_rope = q.split([nope, rdim], dim=-1)

        compressed = self.kv_a_proj(x)
        c_kv, k_rope = compressed.split([self.kv_lora_rank, rdim], dim=-1)
        k_rope = k_rope.view(B, S, 1, rdim).transpose(1, 2)  # (B, 1, S, rdim), shared
        kv = self.kv_b_proj(self.kv_a_norm(c_kv)).view(B, S, H, nope + vdim)
        kv = kv.transpose(1, 2)
        k_nope, v = kv.split([nope, vdim], dim=-1)

        if rope is not None:
            assert position_ids is not None, (
                "position_ids cannot be None when using RoPE"
            )
            q_rope = rope(q_rope, position_ids)
            k_rope = rope(k_rope, position_ids)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(B, H, S, rdim)], dim=-1)

        out = self._attn_fn(q, k, v, attn_mask)
        out = out.transpose(1, 2).reshape(B, S, H * vdim)
        return self.attn_dropout(self.o_proj(out))


ATTN_REGISTRY = {
    "mha": MultiHeadAttention,
    "gqa": GroupedQueryAttention,
    "mla": MultiHeadLatentAttention,
}
