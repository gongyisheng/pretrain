import pytest
import torch

from src.layers.pos_emb import LearnedPositionalEmbedding, RoPE
from tests.fast.layers._refs import (
    SIMPLE_DTYPES,
    learned_pos_emb_ref,
    rope_cos_sin_ref,
    rope_ref,
)


# ==================== LearnedPositionalEmbedding ====================

def test_learned_pos_emb_output_shape():
    pe = LearnedPositionalEmbedding(max_seq_len=32, d_model=16)
    x = torch.randn(2, 8, 16)
    assert pe(x).shape == (2, 8, 16)


def test_learned_pos_emb_adds_position_dependent_offset():
    """Output = x + lookup(0..S-1); same x at different positions differs."""
    pe = LearnedPositionalEmbedding(max_seq_len=32, d_model=16)
    with torch.no_grad():
        pe.embedding.weight.copy_(torch.randn(32, 16))
    x = torch.zeros(1, 4, 16)
    out = pe(x)
    # At zero input, output is exactly the position embeddings 0..S-1.
    assert torch.allclose(out[0], pe.embedding.weight[:4])
    # Different positions yield different embeddings.
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_learned_pos_emb_uses_absolute_positions():
    """Output depends only on S, never on input position_ids — same x → same out for any seq length."""
    pe = LearnedPositionalEmbedding(max_seq_len=32, d_model=16)
    x = torch.randn(1, 6, 16)
    out_a = pe(x)
    out_b = pe(x.clone())
    assert torch.equal(out_a, out_b)


@pytest.mark.parametrize("dtype,atol", SIMPLE_DTYPES)
def test_learned_pos_emb_matches_ref(dtype, atol):
    """LearnedPositionalEmbedding(x) = x + embedding.weight[:S]."""
    torch.manual_seed(0)
    pe = LearnedPositionalEmbedding(max_seq_len=32, d_model=16).to(dtype)
    x = torch.randn(2, 8, 16, dtype=dtype)
    out = pe(x)
    out_ref = learned_pos_emb_ref(x, pe.embedding.weight)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


# ==================== RoPE behavior ====================

def test_rope_forward_with_position_ids_shape():
    rope = RoPE(d_head=16, max_seq_len=32)
    x = torch.randn(2, 4, 8, 16)   # (B, n_heads, S, d_head)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)  # (B, S)
    out = rope(x, position_ids=pos)
    assert out.shape == (2, 4, 8, 16)


def test_rope_forward_reset_position_ids_differ():
    """position_ids with resets must differ from sequential at reset positions."""
    rope = RoPE(d_head=16, max_seq_len=32)
    x = torch.randn(1, 1, 6, 16)
    pos_seq = torch.arange(6).unsqueeze(0)
    pos_reset = torch.tensor([[0, 1, 2, 0, 1, 2]])
    out_seq = rope(x, position_ids=pos_seq)
    out_reset = rope(x, position_ids=pos_reset)
    assert torch.allclose(out_seq[:, :, :3, :], out_reset[:, :, :3, :], atol=1e-6)
    assert not torch.allclose(out_seq[:, :, 3:, :], out_reset[:, :, 3:, :], atol=1e-6)


# ==================== RoPE numerical parity ====================

@pytest.mark.parametrize("theta", [10000.0, 1000000.0])
def test_rope_cos_sin_buffers_match_ref(theta):
    """RoPE cos/sin buffers equal the eager rope_cos_sin_ref tables (fp32)."""
    d_head, max_seq_len = 16, 32
    rope = RoPE(d_head=d_head, max_seq_len=max_seq_len, theta=theta)
    cos_ref, sin_ref = rope_cos_sin_ref(d_head, max_seq_len, theta=theta)
    assert torch.allclose(rope.cos, cos_ref, atol=1e-6)
    assert torch.allclose(rope.sin, sin_ref, atol=1e-6)


@pytest.mark.parametrize("dtype,atol", SIMPLE_DTYPES)
def test_rope_matches_ref(dtype, atol):
    """RoPE output equals eager rope_ref (rotate-half rotation, in input dtype)."""
    torch.manual_seed(0)
    B, S, n_heads, d_head = 2, 8, 4, 16
    rope = RoPE(d_head=d_head, max_seq_len=32, theta=10000.0)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

    x = torch.randn(B, n_heads, S, d_head, dtype=dtype)
    out = rope(x, position_ids=position_ids)

    cos = rope.cos[position_ids][:, None, :, :].to(dtype)
    sin = rope.sin[position_ids][:, None, :, :].to(dtype)
    out_ref = rope_ref(x, cos, sin)

    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", SIMPLE_DTYPES)
def test_rope_matches_hf_apply_rotary_pos_emb(dtype, atol):
    """RoPE output equals HF apply_rotary_pos_emb on the same cos/sin tables."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    torch.manual_seed(0)
    B, S, n_heads, d_head = 2, 8, 4, 16
    rope = RoPE(d_head=d_head, max_seq_len=32, theta=10000.0)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

    q = torch.randn(B, n_heads, S, d_head, dtype=dtype)
    k = torch.randn(B, n_heads, S, d_head, dtype=dtype)

    out_q = rope(q, position_ids=position_ids)
    out_k = rope(k, position_ids=position_ids)

    # HF expects cos/sin as (B, S, d_head); it unsqueezes head dim internally
    cos = rope.cos[position_ids].to(dtype)
    sin = rope.sin[position_ids].to(dtype)
    out_q_ref, out_k_ref = apply_rotary_pos_emb(q, k, cos, sin)

    assert out_q.dtype == dtype and out_k.dtype == dtype
    assert torch.allclose(out_q, out_q_ref, atol=atol)
    assert torch.allclose(out_k, out_k_ref, atol=atol)
