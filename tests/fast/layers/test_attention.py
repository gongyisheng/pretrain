"""
Attention tests: F.scaled_dot_product_attention vs sdpa_ref, plus MHA/GQA
module parity against an eager spec built from sdpa_ref.
"""
import pytest
import torch
import torch.nn.functional as F

from src.layers.attention import GroupedQueryAttention, MultiHeadAttention
from src.layers.rope import RoPE
from src.utils.masking_utils import build_causal_mask
from tests.fast.layers._refs import gqa_ref, mha_ref, sdpa_ref


# ====================== F.scaled_dot_product_attention vs sdpa_ref ======================
# Pins down attention's math (q·k/√d → mask → softmax → ·v) so a future change
# in the SDPA backend (flash, mem-efficient, math) that drifts from the spec
# gets caught.

def _make_qkv(B, H, S, D, dtype):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, H, S, D, dtype=dtype, generator=g)
    k = torch.randn(B, H, S, D, dtype=dtype, generator=g)
    v = torch.randn(B, H, S, D, dtype=dtype, generator=g)
    return q, k, v


SDPA_DTYPES = [
    (torch.float32, 1e-5),
    (torch.float16, 5e-3),
    (torch.bfloat16, 5e-2),
]


@pytest.mark.parametrize("shape", [(2, 4, 8, 16), (1, 8, 32, 32)])
@pytest.mark.parametrize("dtype,atol", SDPA_DTYPES)
def test_sdpa_matches_ref_no_mask(shape, dtype, atol):
    B, H, S, D = shape
    q, k, v = _make_qkv(B, H, S, D, dtype)
    out = F.scaled_dot_product_attention(q, k, v)
    out_ref = sdpa_ref(q, k, v)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", SDPA_DTYPES)
def test_sdpa_matches_ref_is_causal(dtype, atol):
    q, k, v = _make_qkv(2, 4, 8, 16, dtype)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_ref = sdpa_ref(q, k, v, is_causal=True)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", SDPA_DTYPES)
def test_sdpa_matches_ref_additive_mask(dtype, atol):
    """Document-packed causal mask via build_causal_mask."""
    B, H, S, D = 1, 4, 4, 16
    q, k, v = _make_qkv(B, H, S, D, dtype)
    pos = torch.tensor([[0, 1, 0, 1]])  # two docs
    mask = build_causal_mask(pos, device=q.device, dtype=q.dtype)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out_ref = sdpa_ref(q, k, v, attn_mask=mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", SDPA_DTYPES)
def test_sdpa_matches_ref_custom_scale(dtype, atol):
    q, k, v = _make_qkv(2, 4, 8, 16, dtype)
    scale = 0.25
    out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    out_ref = sdpa_ref(q, k, v, scale=scale)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_softmax_fp32_no_overflow(dtype):
    """Large q,k: post-scale logits beyond exp()'s representable range in input dtype.
    Without fp32 softmax, exp() would overflow (fp16: >11, bf16: >88).
    """
    q, k, v = _make_qkv(2, 4, 8, 64, dtype)
    q = q * 30
    k = k * 30
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_ref = sdpa_ref(q, k, v, is_causal=True)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, atol=5e-3)


# ============================= MultiHeadAttention behavior =============================

def test_mha_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0)
    x = torch.randn(2, 16, 64)
    assert mha(x).shape == (2, 16, 64)


def test_mha_is_causal():
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
    assert mha(x, attn_mask=attn_mask).shape == (2, 8, 64)


def test_mha_attn_mask_blocks_cross_doc_attention():
    """Token in doc1 must not be influenced by tokens in doc0."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0)
    mha.eval()

    x = torch.randn(1, 4, 64)
    pos = torch.tensor([[0, 1, 0, 1]])  # doc0=[pos0,pos1], doc1=[pos2,pos3]
    attn_mask = build_causal_mask(pos, device=x.device, dtype=x.dtype)

    out_base = mha(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64)
    x2[0, 1, :] = torch.randn(64)
    out_modified = mha(x2, attn_mask=attn_mask)

    assert torch.allclose(out_base[0, 2:], out_modified[0, 2:], atol=1e-5)
    assert not torch.allclose(out_base[0, :2], out_modified[0, :2], atol=1e-5)


# ============================= GroupedQueryAttention behavior =============================

def test_gqa_output_shape():
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0)
    x = torch.randn(2, 16, 64)
    assert gqa(x).shape == (2, 16, 64)


def test_gqa_is_causal():
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0)
    gqa.eval()
    x = torch.randn(1, 8, 64)
    out_full = gqa(x)
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = gqa(x2)
    assert torch.allclose(out_full[0, :7], out_modified[0, :7], atol=1e-6)


def test_gqa_attn_mask_output_shape():
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask = build_causal_mask(pos, device=x.device, dtype=x.dtype)
    assert gqa(x, attn_mask=attn_mask).shape == (2, 8, 64)


def test_gqa_attn_mask_blocks_cross_doc_attention():
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0)
    gqa.eval()

    x = torch.randn(1, 4, 64)
    pos = torch.tensor([[0, 1, 0, 1]])
    attn_mask = build_causal_mask(pos, device=x.device, dtype=x.dtype)

    out_base = gqa(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64)
    x2[0, 1, :] = torch.randn(64)
    out_modified = gqa(x2, attn_mask=attn_mask)

    assert torch.allclose(out_base[0, 2:], out_modified[0, 2:], atol=1e-5)
    assert not torch.allclose(out_base[0, :2], out_modified[0, :2], atol=1e-5)


def test_gqa_with_rope_output_shape():
    rope = RoPE(d_head=16, max_seq_len=32)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    assert gqa(x, rope, position_ids=pos).shape == (2, 8, 64)


# ============================= MHA / GQA numerical parity vs eager ref =============================

def _mha_ref_call(mha, x, attn_mask=None):
    return mha_ref(
        x, mha.q_proj, mha.k_proj, mha.v_proj, mha.o_proj, mha.n_heads,
        q_norm=getattr(mha, "q_norm", None),
        k_norm=getattr(mha, "k_norm", None),
        attn_mask=attn_mask,
    )


def _gqa_ref_call(gqa, x, attn_mask=None):
    return gqa_ref(
        x, gqa.q_proj, gqa.k_proj, gqa.v_proj, gqa.o_proj, gqa.n_heads, gqa.n_kv_heads,
        q_norm=getattr(gqa, "q_norm", None),
        k_norm=getattr(gqa, "k_norm", None),
        attn_mask=attn_mask,
    )


# MHA/GQA carry an extra o_proj GEMM after SDPA, so the SDPA-level X5 tolerance
# gets amplified slightly. Loosen fp16 from 5e-3 → 1e-2; bf16 5e-2 is unchanged.
MODULE_DTYPES = [
    (torch.float32, 1e-5),
    (torch.float16, 1e-2),
    (torch.bfloat16, 5e-2),
]


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_mha_matches_ref_causal(dtype, atol):
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0).to(dtype)
    mha.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    out = mha(x)
    out_ref = _mha_ref_call(mha, x)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_mha_matches_ref_attn_mask(dtype, atol):
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0).to(dtype)
    mha.eval()
    x = torch.randn(1, 4, 64, dtype=dtype)
    pos = torch.tensor([[0, 1, 0, 1]])
    mask = build_causal_mask(pos, device=x.device, dtype=x.dtype)
    out = mha(x, attn_mask=mask)
    out_ref = _mha_ref_call(mha, x, attn_mask=mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_gqa_matches_ref_causal(dtype, atol):
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0).to(dtype)
    gqa.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    out = gqa(x)
    out_ref = _gqa_ref_call(gqa, x)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_gqa_matches_ref_attn_mask(dtype, atol):
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0).to(dtype)
    gqa.eval()
    x = torch.randn(1, 4, 64, dtype=dtype)
    pos = torch.tensor([[0, 1, 0, 1]])
    mask = build_causal_mask(pos, device=x.device, dtype=x.dtype)
    out = gqa(x, attn_mask=mask)
    out_ref = _gqa_ref_call(gqa, x, attn_mask=mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


# --- qk_norm parity (Qwen3-style: RMSNorm Q and K before SDPA) ---

@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_mha_qk_norm_matches_ref(dtype, atol):
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, qk_norm=True).to(dtype)
    mha.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    out = mha(x)
    out_ref = _mha_ref_call(mha, x)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_gqa_qk_norm_matches_ref(dtype, atol):
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, qk_norm=True).to(dtype)
    gqa.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    out = gqa(x)
    out_ref = _gqa_ref_call(gqa, x)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)
