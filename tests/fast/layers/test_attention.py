import torch
import torch.nn.functional as F

from src.layers.attention import GroupedQueryAttention, MultiHeadAttention
from src.layers.rope import RoPE
from src.utils.masking_utils import build_causal_mask


# --- MultiHeadAttention ---

def test_multihead_attention_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0)
    x = torch.randn(2, 16, 64)
    out = mha(x)
    assert out.shape == (2, 16, 64)


def test_multihead_attention_is_causal():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0)
    mha.eval()
    x = torch.randn(1, 8, 64)
    out_full = mha(x)
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = mha(x2)
    assert torch.allclose(out_full[0, :7], out_modified[0, :7], atol=1e-6)


def test_mha_attn_mask_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask = build_causal_mask(pos, device=x.device, dtype=x.dtype)
    out = mha(x, attn_mask=attn_mask)
    assert out.shape == (2, 8, 64)


def test_mha_attn_mask_blocks_cross_doc_attention():
    """Token in doc1 must not be influenced by tokens in doc0."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0)
    mha.eval()

    x = torch.randn(1, 4, 64)
    # position_ids [0,1,0,1]: doc0=[pos0,pos1], doc1=[pos2,pos3]
    pos = torch.tensor([[0, 1, 0, 1]])
    attn_mask = build_causal_mask(pos, device=x.device, dtype=x.dtype)

    out_base = mha(x, attn_mask=attn_mask)

    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64)
    x2[0, 1, :] = torch.randn(64)
    out_modified = mha(x2, attn_mask=attn_mask)

    assert torch.allclose(out_base[0, 2:], out_modified[0, 2:], atol=1e-5), \
        "doc1 tokens were affected by changes to doc0 tokens"


# --- GroupedQueryAttention ---

def test_gqa_forward_with_position_ids_shape():
    """GroupedQueryAttention accepts position_ids and returns correct shape."""
    rope = RoPE(d_head=16, max_seq_len=32)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    out = gqa(x, rope, position_ids=pos)
    assert out.shape == (2, 8, 64)


# --- Numerical parity vs F.scaled_dot_product_attention ---

def test_mha_matches_sdpa_reference():
    """MHA output equals a manually-composed F.scaled_dot_product_attention reference."""
    torch.manual_seed(0)
    B, S, d_model, n_heads = 2, 8, 64, 4
    d_head = d_model // n_heads
    mha = MultiHeadAttention(d_model, n_heads, dropout_attn=0.0)
    mha.eval()
    x = torch.randn(B, S, d_model)

    # Manual reference using the same weights
    q = mha.q_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    k = mha.k_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    v = mha.v_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / (d_head**0.5))
    ref = mha.o_proj(attn.transpose(1, 2).reshape(B, S, d_model))

    assert torch.allclose(mha(x), ref, atol=1e-5)


def test_gqa_matches_sdpa_with_kv_expansion_reference():
    """GQA output equals SDPA with KV heads expanded via repeat_interleave.

    Catches divergence between our expand+reshape expansion and the
    canonical repeat_interleave form.
    """
    torch.manual_seed(0)
    B, S, d_model, n_heads, n_kv_heads = 2, 8, 64, 4, 2
    d_head = d_model // n_heads
    n_groups = n_heads // n_kv_heads
    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_attn=0.0)
    gqa.eval()
    x = torch.randn(B, S, d_model)

    q = gqa.q_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    k = gqa.k_proj(x).reshape(B, S, n_kv_heads, d_head).transpose(1, 2)
    v = gqa.v_proj(x).reshape(B, S, n_kv_heads, d_head).transpose(1, 2)
    k = k.repeat_interleave(n_groups, dim=1)
    v = v.repeat_interleave(n_groups, dim=1)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / (d_head**0.5))
    ref = gqa.o_proj(attn.transpose(1, 2).reshape(B, S, d_model))

    assert torch.allclose(gqa(x), ref, atol=1e-5)


def test_mha_attn_mask_matches_sdpa_reference():
    """With an explicit attn_mask (no is_causal), MHA matches SDPA reference."""
    torch.manual_seed(0)
    B, S, d_model, n_heads = 1, 4, 64, 4
    d_head = d_model // n_heads
    mha = MultiHeadAttention(d_model, n_heads, dropout_attn=0.0)
    mha.eval()
    x = torch.randn(B, S, d_model)
    pos = torch.tensor([[0, 1, 0, 1]])  # two docs
    attn_mask = build_causal_mask(pos, device=x.device, dtype=x.dtype)

    q = mha.q_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    k = mha.k_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    v = mha.v_proj(x).reshape(B, S, n_heads, d_head).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, scale=1.0 / (d_head**0.5))
    ref = mha.o_proj(attn.transpose(1, 2).reshape(B, S, d_model))

    assert torch.allclose(mha(x, attn_mask=attn_mask), ref, atol=1e-5)
