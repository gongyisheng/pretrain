"""
Eager reference implementations for layer-level numerical parity tests.
"""
import math

import torch


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

def relu_ref(x: torch.Tensor) -> torch.Tensor:
    """relu(x) = max(x, 0)."""
    return torch.where(x > 0, x, torch.zeros_like(x))


def gelu_ref(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))). Matches F.gelu(approximate='none')."""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu_ref(x: torch.Tensor) -> torch.Tensor:
    """silu(x) = x * sigmoid(x)."""
    return x * torch.sigmoid(x)


UNARY_REFS = {"relu": relu_ref, "gelu": gelu_ref, "silu": silu_ref}


def glu_ref(name: str, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Gated activation: act(gate) * up."""
    return UNARY_REFS[name](gate) * up


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
