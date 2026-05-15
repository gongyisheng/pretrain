"""
Eager reference implementations for layer-level numerical parity tests.
"""
import math

import torch
import torch.nn as nn


# Per-dtype atol. SIMPLE = elementwise/reduction, COMPOUND = GEMM+softmax stacks.
# 1 ULP at output magnitude |x| = 2^(-mantissa_bits) × |x|; here we use |x|~1.
# fp16/bf16 follow ULP multipliers (5× SIMPLE, 10× COMPOUND);
# fp32 uses the conventional 1e-5 noise floor (kernel drift dominates over ULP).
#   fp32 (23-bit mantissa, 1 ULP ≈ 1e-7):  SIMPLE 1e-5 (80×),  COMPOUND 1e-5 (80×)
#   fp16 (10-bit mantissa, 1 ULP ≈ 1e-3):  SIMPLE 5e-3  (5×),  COMPOUND 1e-2 (10×)
#   bf16 ( 7-bit mantissa, 1 ULP ≈ 8e-3):  SIMPLE 4e-2  (5×),  COMPOUND 8e-2 (10×)
SIMPLE_DTYPES = [
    (torch.float32, 1e-5),
    (torch.float16, 5e-3),
    (torch.bfloat16, 4e-2),
]
COMPOUND_DTYPES = [
    (torch.float32, 1e-5),
    (torch.float16, 1e-2),
    (torch.bfloat16, 8e-2),
]


def rmsnorm_ref(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    eps: float
) -> torch.Tensor:
    """y = weight * x / sqrt(mean(x^2) + eps), reduction in fp32."""
    dtype = x.dtype
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * out.to(dtype)


def layernorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """y = weight * (x - mean) / sqrt(var + eps) + bias, reduction in fp32."""
    dtype = x.dtype
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, keepdim=True, unbiased=False)
    out = (xf - mean) * torch.rsqrt(var + eps)
    out = weight * out.to(dtype)
    if bias is not None:
        out = out + bias
    return out


# ---------------------------- Activations ----------------------------

# --- Ungated (unary): x → act(x) ---

def relu_ref(x: torch.Tensor) -> torch.Tensor:
    """relu(x) = max(x, 0)."""
    return torch.where(x > 0, x, torch.zeros_like(x))


def gelu_ref(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))). Matches F.gelu(approximate='none')."""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu_ref(x: torch.Tensor) -> torch.Tensor:
    """silu(x) = x * sigmoid(x)."""
    return x * torch.sigmoid(x)


# --- Gated (GLU family): (gate, up) → act(gate) * up ---

def relu_glu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return relu_ref(gate) * up


def gelu_glu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gelu_ref(gate) * up


def silu_glu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return silu_ref(gate) * up


UNGATED_ACTIVATIONS_REFS = {"relu": relu_ref, "gelu": gelu_ref, "silu": silu_ref}
GATED_ACTIVATIONS_REFS = {"relu": relu_glu_ref, "gelu": gelu_glu_ref, "silu": silu_glu_ref}


# ---------------------------- FFN ----------------------------

def ffn_ref(
    x: torch.Tensor,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    activation: str,
    gate_proj: nn.Linear | None = None,
) -> torch.Tensor:
    """Eager feed-forward:
        ungated (gate_proj=None): down_proj(act(up_proj(x)))
        gated   (gate_proj given): down_proj(act(gate_proj(x), up_proj(x)))
    `activation` is a key in {U,G}NGATED_ACTIVATIONS_REFS ("relu"/"gelu"/"silu").
    """
    if gate_proj is not None:
        hidden = GATED_ACTIVATIONS_REFS[activation](gate_proj(x), up_proj(x))
    else:
        hidden = UNGATED_ACTIVATIONS_REFS[activation](up_proj(x))
    return down_proj(hidden)


# ---------------------------- RoPE ----------------------------

def rope_cos_sin_ref(
    d_head: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager RoPE cos/sin tables.

    inv_freq[k] = 1 / theta^(2k / d_head) for k in [0, d_head/2).
    angles[p, k] = p * inv_freq[k]; concatenated along the last dim so
    rotate_half pairs the right components.
    Returns (cos, sin), each shape (max_seq_len, d_head), fp32.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return torch.cos(angles), torch.sin(angles)


def rope_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Eager rotary position embedding: x * cos + rotate_half(x) * sin.

    rotate_half(x): split x last-dim into halves, return concat([-x2, x1]).
    x: (..., d_head); cos/sin: broadcastable to x's shape, in input dtype.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# ---------------------------- Attention ----------------------------

def sdpa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Eager scaled-dot-product attention, matching Flash Attention's precision pattern.

    GEMMs stay in input dtype (tensor cores accumulate in fp32 internally, output cast
    back); only the softmax runs explicitly in fp32. Mirrors F.scaled_dot_product_attention:
    attn_mask is additive (-inf for masked, 0 for allowed). Inputs (..., S, D).
    """
    dtype = q.dtype
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    logits = (q @ k.transpose(-1, -2)) * scale      # input dtype
    if is_causal:
        Sq, Sk = q.shape[-2], k.shape[-2]
        causal = torch.ones(Sq, Sk, dtype=torch.bool, device=q.device).tril()
        logits = logits.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        logits = logits + attn_mask.to(logits.dtype)
    attn = logits.float().softmax(dim=-1).to(dtype)  # upcast only for softmax, cast back before second GEMM
    return attn @ v


def mha_ref(
    x: torch.Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    o_proj: nn.Linear,
    n_heads: int,
    q_norm: nn.Module | None = None,
    k_norm: nn.Module | None = None,
    rope: nn.Module | None = None,
    position_ids: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager MHA: q/k/v projections → optional qk_norm → optional rope → sdpa_ref → o_proj."""
    B, S, _ = x.shape
    d_head = q_proj.out_features // n_heads
    q = q_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    k = k_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    v = v_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    if q_norm is not None:
        q = q_norm(q)
        k = k_norm(k)
    if rope is not None:
        q = rope(q, position_ids=position_ids)
        k = rope(k, position_ids=position_ids)
    attn = sdpa_ref(q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None))
    return o_proj(attn.transpose(1, 2).reshape(B, S, n_heads * d_head))


def gqa_ref(
    x: torch.Tensor,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    o_proj: nn.Linear,
    n_heads: int,
    n_kv_heads: int,
    q_norm: nn.Module | None = None,
    k_norm: nn.Module | None = None,
    rope: nn.Module | None = None,
    position_ids: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager GQA: GQA-shaped projections → optional qk_norm → optional rope → KV expansion → sdpa_ref → o_proj."""
    B, S, _ = x.shape
    d_head = q_proj.out_features // n_heads
    n_groups = n_heads // n_kv_heads
    q = q_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    k = k_proj(x).reshape(B, S, n_kv_heads, d_head).transpose(1, 2)
    v = v_proj(x).reshape(B, S, n_kv_heads, d_head).transpose(1, 2)
    if q_norm is not None:
        q = q_norm(q)
        k = k_norm(k)
    if rope is not None:
        q = rope(q, position_ids=position_ids)
        k = rope(k, position_ids=position_ids)
    k = k.repeat_interleave(n_groups, dim=1)
    v = v.repeat_interleave(n_groups, dim=1)
    attn = sdpa_ref(q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None))
    return o_proj(attn.transpose(1, 2).reshape(B, S, n_heads * d_head))
