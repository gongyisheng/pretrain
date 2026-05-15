"""
Attention tests: F.scaled_dot_product_attention vs sdpa_ref, plus MHA/GQA
module parity against an eager spec built from sdpa_ref.
"""
import pytest
import torch
import torch.nn.functional as F

from src.layers.attention import GroupedQueryAttention, MultiHeadAttention
from src.layers.pos_emb import RoPE
from src.utils.masking_utils import build_attention_mask
from tests.fast.layers._refs import COMPOUND_DTYPES, gqa_ref, mha_ref, sdpa_ref


# Attention backends to test. flex_attention is CUDA-only, so it's wrapped in
# skipif. Each entry pairs the implementation name with the device its kernel
# runs on; tests parametrized over this iterate the cartesian product of
# backend × shape/dtype configurations.
IMPL_DEVICE = [
    pytest.param("sdpa", "cpu", id="sdpa-cpu"),
    pytest.param(
        "flex_attention",
        "cuda",
        id="flex-cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA"),
    ),
]


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
    """Document-packed causal mask via build_attention_mask (sdpa form)."""
    B, H, S, D = 1, 4, 4, 16
    q, k, v = _make_qkv(B, H, S, D, dtype)
    pos = torch.tensor([[0, 1, 0, 1]])  # two docs
    mask = build_attention_mask(pos, q.device, q.dtype, attn_implementation="sdpa")
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


@pytest.mark.parametrize("dtype,atol", [(torch.float16, 5e-2), (torch.bfloat16, 5e-1)])
def test_sdpa_softmax_fp32_no_overflow(dtype, atol):
    """Large q,k: post-scale logits beyond exp()'s representable range in input dtype.
    Without fp32 softmax, exp() would overflow (fp16: >11, bf16: >88) and produce
    NaN/Inf. atol is generous: at this scale softmax saturates near winner-take-all,
    so flash-attn's online softmax and the reference's plain softmax can pick
    different argmax winners. bf16's coarser mantissa amplifies this.
    """
    q, k, v = _make_qkv(2, 4, 8, 64, dtype)
    q = q * 30
    k = k * 30
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_ref = sdpa_ref(q, k, v, is_causal=True)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_ref, atol=atol)


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


@pytest.mark.parametrize("impl,device", IMPL_DEVICE)
def test_mha_attn_mask_output_shape(impl, device):
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation=impl).to(device)
    x = torch.randn(2, 8, 64, device=device)
    pos = torch.arange(8, device=device).unsqueeze(0).expand(2, -1)
    attn_mask = build_attention_mask(pos, x.device, x.dtype, attn_implementation=impl)
    assert mha(x, attn_mask=attn_mask).shape == (2, 8, 64)


@pytest.mark.parametrize("impl,device", IMPL_DEVICE)
def test_mha_attn_mask_blocks_cross_doc_attention(impl, device):
    """Token in doc1 must not be influenced by tokens in doc0."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation=impl).to(device)
    mha.eval()

    x = torch.randn(1, 4, 64, device=device)
    pos = torch.tensor([[0, 1, 0, 1]], device=device)  # doc0=[pos0,pos1], doc1=[pos2,pos3]
    attn_mask = build_attention_mask(pos, x.device, x.dtype, attn_implementation=impl)

    out_base = mha(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64, device=device)
    x2[0, 1, :] = torch.randn(64, device=device)
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


@pytest.mark.parametrize("impl,device", IMPL_DEVICE)
def test_gqa_attn_mask_output_shape(impl, device):
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation=impl).to(device)
    x = torch.randn(2, 8, 64, device=device)
    pos = torch.arange(8, device=device).unsqueeze(0).expand(2, -1)
    attn_mask = build_attention_mask(pos, x.device, x.dtype, attn_implementation=impl)
    assert gqa(x, attn_mask=attn_mask).shape == (2, 8, 64)


@pytest.mark.parametrize("impl,device", IMPL_DEVICE)
def test_gqa_attn_mask_blocks_cross_doc_attention(impl, device):
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation=impl).to(device)
    gqa.eval()

    x = torch.randn(1, 4, 64, device=device)
    pos = torch.tensor([[0, 1, 0, 1]], device=device)
    attn_mask = build_attention_mask(pos, x.device, x.dtype, attn_implementation=impl)

    out_base = gqa(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64, device=device)
    x2[0, 1, :] = torch.randn(64, device=device)
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


MODULE_DTYPES = COMPOUND_DTYPES


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
@pytest.mark.parametrize("impl,device", IMPL_DEVICE)
def test_mha_matches_ref_attn_mask(impl, device, dtype, atol):
    # FlexAttention doesn't support fp16/bf16 storage of the BlockMask machinery
    # on this device for these tiny shapes — keep numerical parity at fp32.
    if impl == "flex_attention" and dtype != torch.float32:
        pytest.skip("flex_attention reference parity only checked at fp32")
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation=impl).to(device).to(dtype)
    mha.eval()
    x = torch.randn(1, 4, 64, dtype=dtype, device=device)
    pos = torch.tensor([[0, 1, 0, 1]], device=device)
    attn_mask = build_attention_mask(pos, x.device, x.dtype, attn_implementation=impl)
    # Reference always uses the dense mask shape (manual masked softmax).
    dense_mask = build_attention_mask(pos, x.device, x.dtype, attn_implementation="sdpa")
    out = mha(x, attn_mask=attn_mask)
    out_ref = _mha_ref_call(mha, x, attn_mask=dense_mask)
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
@pytest.mark.parametrize("impl,device", IMPL_DEVICE)
def test_gqa_matches_ref_attn_mask(impl, device, dtype, atol):
    if impl == "flex_attention" and dtype != torch.float32:
        pytest.skip("flex_attention reference parity only checked at fp32")
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation=impl).to(device).to(dtype)
    gqa.eval()
    x = torch.randn(1, 4, 64, dtype=dtype, device=device)
    pos = torch.tensor([[0, 1, 0, 1]], device=device)
    attn_mask = build_attention_mask(pos, x.device, x.dtype, attn_implementation=impl)
    dense_mask = build_attention_mask(pos, x.device, x.dtype, attn_implementation="sdpa")
    out = gqa(x, attn_mask=attn_mask)
    out_ref = _gqa_ref_call(gqa, x, attn_mask=dense_mask)
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


# ====================== Cross-backend equivalence ======================
# The sdpa and flex_attention paths must produce numerically equivalent
# attention output for the same logical doc-causal mask — otherwise we'd
# silently change attention semantics when switching attn_implementation.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_mha_sdpa_and_flex_match():
    torch.manual_seed(0)
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device="cuda")
    x = torch.randn(1, 8, 64, device="cuda")

    mha_sdpa = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation="sdpa").cuda().eval()
    mha_flex = MultiHeadAttention(d_model=64, n_heads=4, dropout_attn=0.0, attn_implementation="flex_attention").cuda().eval()
    # Share weights so we're only comparing the attention kernel, not init noise.
    mha_flex.load_state_dict(mha_sdpa.state_dict())

    out_sdpa = mha_sdpa(x, attn_mask=build_attention_mask(pos, x.device, x.dtype, attn_implementation="sdpa"))
    out_flex = mha_flex(x, attn_mask=build_attention_mask(pos, x.device, x.dtype, attn_implementation="flex_attention"))
    assert torch.allclose(out_sdpa, out_flex, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_gqa_sdpa_and_flex_match():
    torch.manual_seed(0)
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device="cuda")
    x = torch.randn(1, 8, 64, device="cuda")

    gqa_sdpa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation="sdpa").cuda().eval()
    gqa_flex = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout_attn=0.0, attn_implementation="flex_attention").cuda().eval()
    gqa_flex.load_state_dict(gqa_sdpa.state_dict())

    out_sdpa = gqa_sdpa(x, attn_mask=build_attention_mask(pos, x.device, x.dtype, attn_implementation="sdpa"))
    out_flex = gqa_flex(x, attn_mask=build_attention_mask(pos, x.device, x.dtype, attn_implementation="flex_attention"))
    assert torch.allclose(out_sdpa, out_flex, atol=1e-5)
