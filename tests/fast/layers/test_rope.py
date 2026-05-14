import torch

from src.layers.rope import RoPE


# --- Numerical parity vs HuggingFace apply_rotary_pos_emb ---

def test_rope_matches_hf_apply_rotary_pos_emb():
    """RoPE output equals HF apply_rotary_pos_emb on the same cos/sin tables."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    torch.manual_seed(0)
    B, S, n_heads, d_head = 2, 8, 4, 16
    rope = RoPE(d_head=d_head, max_seq_len=32, theta=10000.0)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

    q = torch.randn(B, n_heads, S, d_head)
    k = torch.randn(B, n_heads, S, d_head)

    our_q = rope(q, position_ids=position_ids)
    our_k = rope(k, position_ids=position_ids)

    # HF expects cos/sin as (B, S, d_head); it unsqueezes head dim internally
    cos = rope.cos[position_ids]
    sin = rope.sin[position_ids]
    hf_q, hf_k = apply_rotary_pos_emb(q, k, cos, sin)

    assert torch.allclose(our_q, hf_q, atol=1e-5)
    assert torch.allclose(our_k, hf_k, atol=1e-5)


# --- Behavioral tests ---

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
