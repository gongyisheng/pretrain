"""
Attention tests: F.scaled_dot_product_attention vs sdpa_ref, plus MHA/GQA
module parity against an eager spec built from sdpa_ref.
"""
import pytest
import torch
import torch.nn.functional as F

from src.layers.attention import GroupedQueryAttention, MultiHeadAttention
from src.layers.pos_emb import RoPE
from src.utils.masking_utils import (
    build_causal_attention_mask,
    build_intra_doc_attention_mask,
)
from tests.fast.layers._refs import COMPOUND_DTYPES, gqa_ref, mha_ref, sdpa_ref


# Attention backends to test. The device comes from the conftest ``device``
# fixture (--device=cpu|cuda, default cuda if available else cpu): one device
# per session, not iterated. flex_attention is CUDA-only — when the session
# is on CPU, the flex_attention parametrization is skipped at runtime.
IMPL = ["sdpa", "flex_attention"]


def _skip_if_unsupported(impl, device):
    if impl == "flex_attention" and device == "cpu":
        pytest.skip("flex_attention requires CUDA")

# Mask shapes the trainer actually builds. "causal" is the pure-causal pattern
# (no doc boundaries); "intra_doc" blocks attention across document boundaries.
MASK_KIND = ["causal", "intra_doc"]


def _make_attn_mask(kind, impl, position_ids, dtype):
    """Build the mask the kernel consumes plus a reference-side dense mask.

    The reference implementations (``mha_ref`` / ``gqa_ref``) compute attention
    manually from a dense additive mask. To compare them apples-to-apples we
    hand them either ``None`` (when the kernel does pure-causal) or the dense
    intra-doc mask — never a ``BlockMask``.
    """
    B, S = position_ids.shape
    if kind == "causal":
        kernel_mask = build_causal_attention_mask(B, S, position_ids.device, attn_implementation=impl)
        ref_mask = None  # ref uses is_causal=True
    elif kind == "intra_doc":
        kernel_mask = build_intra_doc_attention_mask(position_ids, position_ids.device, dtype, attn_implementation=impl)
        ref_mask = build_intra_doc_attention_mask(position_ids, position_ids.device, dtype, attn_implementation="sdpa")
    else:
        raise AssertionError(f"unknown mask kind: {kind!r}")
    return kernel_mask, ref_mask


# ====================== F.scaled_dot_product_attention vs sdpa_ref ======================
# Pins down attention's math (q·k/√d → mask → softmax → ·v) so a future change
# in the SDPA backend (flash, mem-efficient, math) that drifts from the spec
# gets caught.

def _make_qkv(B, H, S, D, dtype):
    g = torch.Generator(device=torch.get_default_device()).manual_seed(0)
    q = torch.randn(B, H, S, D, dtype=dtype, generator=g)
    k = torch.randn(B, H, S, D, dtype=dtype, generator=g)
    v = torch.randn(B, H, S, D, dtype=dtype, generator=g)
    return q, k, v


SDPA_DTYPES = COMPOUND_DTYPES


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
    """Document-packed causal mask via build_intra_doc_attention_mask (sdpa form)."""
    B, H, S, D = 1, 4, 4, 16
    q, k, v = _make_qkv(B, H, S, D, dtype)
    pos = torch.tensor([[0, 1, 0, 1]])  # two docs
    mask = build_intra_doc_attention_mask(pos, q.device, q.dtype, attn_implementation="sdpa")
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


@pytest.mark.parametrize("dtype,atol", [(torch.float16, 5e-3), (torch.bfloat16, 5e-2)])
def test_sdpa_softmax_fp32_no_overflow(dtype, atol):
    """Large q,k: post-scale logits beyond exp()'s representable range in input dtype.
    Without fp32 softmax, exp() would overflow (fp16: >11, bf16: >88) and produce
    NaN/Inf. Both SDPA and ``sdpa_ref`` accumulate in fp32, so at this scale they
    saturate to the same argmax winner — observed gap is one ULP of the storage
    dtype (≈2e-3 fp16, ≈2e-2 bf16). atol is one storage ULP with ~2× margin.
    """
    q, k, v = _make_qkv(2, 4, 8, 64, dtype)
    q = q * 30
    k = k * 30
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_ref = sdpa_ref(q, k, v, is_causal=True)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, atol=atol)


# ============================= MultiHeadAttention behavior =============================

@pytest.mark.parametrize("impl", IMPL)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_mha_attn_mask_output_shape(kind, impl, device):
    _skip_if_unsupported(impl, device)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation=impl)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, _ = _make_attn_mask(kind, impl, pos, x.dtype)
    assert mha(x, attn_mask=attn_mask).shape == (2, 8, 64)


@pytest.mark.parametrize("impl", IMPL)
def test_mha_intra_doc_mask_blocks_cross_doc_attention(impl, device):
    """Token in doc1 must not be influenced by tokens in doc0. Intra-doc only —
    the causal mask doesn't enforce doc boundaries."""
    _skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation=impl)
    mha.eval()

    x = torch.randn(1, 4, 64)
    pos = torch.tensor([[0, 1, 0, 1]])  # doc0=[pos0,pos1], doc1=[pos2,pos3]
    attn_mask, _ = _make_attn_mask("intra_doc", impl, pos, x.dtype)

    out_base = mha(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64)
    x2[0, 1, :] = torch.randn(64)
    out_modified = mha(x2, attn_mask=attn_mask)

    assert torch.allclose(out_base[0, 2:], out_modified[0, 2:], atol=1e-5)
    assert not torch.allclose(out_base[0, :2], out_modified[0, :2], atol=1e-5)


@pytest.mark.parametrize("impl", IMPL)
def test_mha_causal_mask_blocks_future(impl, device):
    """Modifying the last token must not change earlier-token outputs."""
    _skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation=impl)
    mha.eval()

    x = torch.randn(1, 8, 64)
    pos = torch.arange(8).unsqueeze(0)
    attn_mask, _ = _make_attn_mask("causal", impl, pos, x.dtype)

    out_base = mha(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = mha(x2, attn_mask=attn_mask)
    assert torch.allclose(out_base[0, :7], out_modified[0, :7], atol=1e-5)


# ============================= GroupedQueryAttention behavior =============================

@pytest.mark.parametrize("impl", IMPL)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_gqa_attn_mask_output_shape(kind, impl, device):
    _skip_if_unsupported(impl, device)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation=impl)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, _ = _make_attn_mask(kind, impl, pos, x.dtype)
    assert gqa(x, attn_mask=attn_mask).shape == (2, 8, 64)


@pytest.mark.parametrize("impl", IMPL)
def test_gqa_intra_doc_mask_blocks_cross_doc_attention(impl, device):
    _skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation=impl)
    gqa.eval()

    x = torch.randn(1, 4, 64)
    pos = torch.tensor([[0, 1, 0, 1]])
    attn_mask, _ = _make_attn_mask("intra_doc", impl, pos, x.dtype)

    out_base = gqa(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64)
    x2[0, 1, :] = torch.randn(64)
    out_modified = gqa(x2, attn_mask=attn_mask)

    assert torch.allclose(out_base[0, 2:], out_modified[0, 2:], atol=1e-5)
    assert not torch.allclose(out_base[0, :2], out_modified[0, :2], atol=1e-5)


@pytest.mark.parametrize("impl", IMPL)
def test_gqa_causal_mask_blocks_future(impl, device):
    _skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation=impl)
    gqa.eval()

    x = torch.randn(1, 8, 64)
    pos = torch.arange(8).unsqueeze(0)
    attn_mask, _ = _make_attn_mask("causal", impl, pos, x.dtype)

    out_base = gqa(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = gqa(x2, attn_mask=attn_mask)
    assert torch.allclose(out_base[0, :7], out_modified[0, :7], atol=1e-5)


def test_gqa_with_rope_output_shape():
    rope = RoPE(d_head=16, max_seq_len=32)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation="sdpa")
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


MODULE_DTYPES = COMPOUND_DTYPES


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_mha_matches_ref_causal(dtype, atol):
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation="sdpa").to(dtype)
    mha.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    out = mha(x)
    out_ref = _mha_ref_call(mha, x)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
@pytest.mark.parametrize("impl", IMPL)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_mha_matches_ref_attn_mask(kind, impl, device, dtype, atol):
    _skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation=impl).to(dtype)
    mha.eval()
    x = torch.randn(1, 4, 64, dtype=dtype)
    pos = torch.tensor([[0, 1, 0, 1]] if kind == "intra_doc" else [[0, 1, 2, 3]])
    attn_mask, ref_mask = _make_attn_mask(kind, impl, pos, x.dtype)
    out = mha(x, attn_mask=attn_mask)
    out_ref = _mha_ref_call(mha, x, attn_mask=ref_mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_gqa_matches_ref_causal(dtype, atol):
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation="sdpa").to(dtype)
    gqa.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    out = gqa(x)
    out_ref = _gqa_ref_call(gqa, x)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
@pytest.mark.parametrize("impl", IMPL)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_gqa_matches_ref_attn_mask(kind, impl, device, dtype, atol):
    _skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation=impl).to(dtype)
    gqa.eval()
    x = torch.randn(1, 4, 64, dtype=dtype)
    pos = torch.tensor([[0, 1, 0, 1]] if kind == "intra_doc" else [[0, 1, 2, 3]])
    attn_mask, ref_mask = _make_attn_mask(kind, impl, pos, x.dtype)
    out = gqa(x, attn_mask=attn_mask)
    out_ref = _gqa_ref_call(gqa, x, attn_mask=ref_mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


# --- qk_norm parity (Qwen3-style: RMSNorm Q and K before SDPA) ---

@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_mha_qk_norm_matches_ref(dtype, atol):
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, qk_norm=True, attn_implementation="sdpa").to(dtype)
    mha.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    out = mha(x)
    out_ref = _mha_ref_call(mha, x)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
def test_gqa_qk_norm_matches_ref(dtype, atol):
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, qk_norm=True, attn_implementation="sdpa").to(dtype)
    gqa.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    out = gqa(x)
    out_ref = _gqa_ref_call(gqa, x)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


