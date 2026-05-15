"""Numerical parity: F.scaled_dot_product_attention vs eager fp32-softmax reference.

Pins down the math (q·k/√d → mask → softmax → ·v) so a future change in the
SDPA backend (flash, mem-efficient, math) that drifts from the spec gets caught.
"""
import pytest
import torch
import torch.nn.functional as F

from src.utils.masking_utils import build_causal_mask
from tests.fast.layers._refs import sdpa_ref


def _make_qkv(B, H, S, D, dtype):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, H, S, D, dtype=dtype, generator=g)
    k = torch.randn(B, H, S, D, dtype=dtype, generator=g)
    v = torch.randn(B, H, S, D, dtype=dtype, generator=g)
    return q, k, v


@pytest.mark.parametrize("shape", [(2, 4, 8, 16), (1, 8, 32, 32)])
def test_sdpa_matches_ref_no_mask(shape):
    B, H, S, D = shape
    q, k, v = _make_qkv(B, H, S, D, torch.float32)
    out = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, sdpa_ref(q, k, v), atol=1e-5)


def test_sdpa_matches_ref_is_causal():
    q, k, v = _make_qkv(2, 4, 8, 16, torch.float32)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, sdpa_ref(q, k, v, is_causal=True), atol=1e-5)


def test_sdpa_matches_ref_additive_mask():
    """Document-packed causal mask via build_causal_mask."""
    B, H, S, D = 1, 4, 4, 16
    q, k, v = _make_qkv(B, H, S, D, torch.float32)
    pos = torch.tensor([[0, 1, 0, 1]])  # two docs
    mask = build_causal_mask(pos, device=q.device, dtype=q.dtype)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(out, sdpa_ref(q, k, v, attn_mask=mask), atol=1e-5)


def test_sdpa_matches_ref_custom_scale():
    q, k, v = _make_qkv(2, 4, 8, 16, torch.float32)
    scale = 0.25
    out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    assert torch.allclose(out, sdpa_ref(q, k, v, scale=scale), atol=1e-5)


@pytest.mark.parametrize("dtype,atol", [
    (torch.float32, 1e-5),
    (torch.bfloat16, 5e-2),
])
def test_sdpa_dtype_parity_causal(dtype, atol):
    """Causal SDPA matches eager ref within dtype tolerance.

    bf16 tolerance is loose: attention has S-way softmax + S-term sum, so
    error compounds with sequence length and head magnitudes.
    """
    q, k, v = _make_qkv(2, 4, 16, 32, dtype)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.dtype == dtype
    assert torch.allclose(out, sdpa_ref(q, k, v, is_causal=True), atol=atol)


def test_sdpa_softmax_fp32_no_overflow_bf16():
    """Large q,k: q·kᵀ scales to O(D); without fp32 softmax, exp() overflows."""
    q, k, v = _make_qkv(2, 4, 8, 64, torch.bfloat16)
    q = q * 30
    k = k * 30
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.isfinite(out).all()
    # ref also handles it (softmax in fp32); should agree
    assert torch.allclose(out, sdpa_ref(q, k, v, is_causal=True), atol=1e-1)
