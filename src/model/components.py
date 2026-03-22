import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Fused JIT kernels ---
# Each decorated function is compiled by Triton on first call, fusing all
# element-wise ops into a single GPU kernel (avoids redundant memory round-trips).

@torch.compile
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused: fp32 cast → pow2 → mean → rsqrt → scale → cast back."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x.to(dtype)


@torch.compile
def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fused: rotate-half → scale by cos/sin → add."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


@torch.compile
def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused: silu(gate) * up in a single pass."""
    return F.silu(gate) * up


@torch.compile(dynamic=True)
def _attn_res_aggregate_rmsnorm(
    V: torch.Tensor, w_proj: torch.Tensor, norm_weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fused: RMSNorm keys → dot-product logits → softmax → weighted sum.

    Inlining the norm eliminates the K tensor write/read to HBM.
    dynamic=True because N grows across layers.
    """
    dtype = V.dtype
    vf = V.float()
    K = (vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)).to(dtype) * norm_weight
    logits  = (K * w_proj).sum(-1)         # (N, B, S)
    weights = logits.softmax(0)            # (N, B, S)
    return (weights.unsqueeze(-1) * V).sum(0)  # (B, S, D)


@torch.compile(dynamic=True)
def _attn_res_aggregate_layernorm(
    V: torch.Tensor, w_proj: torch.Tensor,
    norm_weight: torch.Tensor, norm_bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fused: LayerNorm keys → dot-product logits → softmax → weighted sum."""
    dtype = V.dtype
    vf   = V.float()
    mean = vf.mean(-1, keepdim=True)
    K    = ((vf - mean) * torch.rsqrt((vf - mean).pow(2).mean(-1, keepdim=True) + eps)).to(dtype) * norm_weight + norm_bias
    logits  = (K * w_proj).sum(-1)
    weights = logits.softmax(0)
    return (weights.unsqueeze(-1) * V).sum(0)


# --- Norms ---

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)


# --- RoPE ---

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, S, d_head)
        S = x.shape[2]
        cos = self.cos[:S][None, None, :, :].to(x.dtype)      # (1, 1, S, d_head)
        sin = self.sin[:S][None, None, :, :].to(x.dtype)      # (1, 1, S, d_head)
        return _apply_rope(x, cos, sin)


# --- Attention ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
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

        # Use PyTorch's scaled_dot_product_attention (flash attention when available)
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
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor, rope: RoPE) -> torch.Tensor:
        B, S, H = x.shape

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)    # (B, n_heads, S, d_head)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2) # (B, n_kv_heads, S, d_head)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2) # (B, n_kv_heads, S, d_head)

        if self.qk_norm:
            q = self.q_norm(q.reshape(-1, S, self.d_head)).view(B, self.n_heads, S, self.d_head)
            k = self.k_norm(k.reshape(-1, S, self.d_head)).view(B, self.n_kv_heads, S, self.d_head)

        q = rope(q)
        k = rope(k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            enable_gqa=True,
        )

        out = out.transpose(1, 2).reshape(B, S, H)
        return self.resid_dropout(self.out_proj(out))


# --- FFN ---

class GeluFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
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
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.down_proj(_swiglu(gate, up))
        x = self.dropout(x)
        return x


# --- Transformer Block ---

def _block_attn_res(
    attn_res_ctx: list,
    x: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    """Compute block-level attention residual.

    Args:
        attn_res_ctx: list of finalized block tensors, each shape (B, S, D)
        x:   current partial block (hidden state), shape (B, S, D)
        proj: Linear(d_model, 1, bias=False) — learned query vector w_l
        norm: RMSNorm or LayerNorm applied to keys before attention

    Returns:
        Attention-weighted combination of all blocks + x, shape (B, S, D)
    """
    V = torch.stack(attn_res_ctx + [x])            # (N+1, B, S, D)
    w = proj.weight.view(-1)
    if isinstance(norm, nn.LayerNorm):
        return _attn_res_aggregate_layernorm(V, w, norm.weight, norm.bias, norm.eps)
    return _attn_res_aggregate_rmsnorm(V, w, norm.weight, norm.eps)

class BaseTransformerBlock(nn.Module):
    """Base transformer block with optional AttnRes support.

    Subclasses implement attn_sublayer() and ffn_sublayer().
    The residual logic (standard or AttnRes) lives here once for all architectures.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = 0,
        attn_res: bool = False,
        attn_res_block_size: int = 1,
        attn_res_norm: str = "rmsnorm",
    ):
        super().__init__()
        self.attn_res = attn_res
        if attn_res:
            self.layer_idx = layer_idx
            self.attn_res_block_size = attn_res_block_size
            norm_cls = nn.LayerNorm if attn_res_norm == "layernorm" else RMSNorm
            self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
            self.attn_res_norm = norm_cls(d_model)
            self.mlp_res_proj = nn.Linear(d_model, 1, bias=False)
            self.mlp_res_norm = norm_cls(d_model)

    def attn_sublayer(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, attn_res_ctx=None, **kwargs) -> tuple:
        if self.attn_res:
            partial_block = x

            # compute h before attn using current partial_block as current-block rep
            h = _block_attn_res(attn_res_ctx, partial_block, self.attn_res_proj, self.attn_res_norm)

            # seal at block boundary before processing attn; seeds blocks with embedding at layer 0
            if self.layer_idx % self.attn_res_block_size == 0:
                attn_res_ctx = attn_res_ctx + [partial_block]
                partial_block = None

            attn_out = self.attn_sublayer(h, **kwargs)
            partial_block = partial_block + attn_out if partial_block is not None else attn_out

            # compute h before FFN
            h = _block_attn_res(attn_res_ctx, partial_block, self.mlp_res_proj, self.mlp_res_norm)

            mlp_out = self.ffn_sublayer(h)
            partial_block = partial_block + mlp_out

            return partial_block, attn_res_ctx
        else:
            # Standard residual path
            x = x + self.attn_sublayer(x, **kwargs)
            x = x + self.ffn_sublayer(x)
            return x

