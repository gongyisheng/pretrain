"""
Attention tests: F.scaled_dot_product_attention vs sdpa_ref, plus MHA/GQA
module parity against an eager spec built from sdpa_ref.
"""

import pytest
import torch
import torch.nn.functional as F

from src.layers.attention import (
    ATTN_REGISTRY,
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiHeadLatentAttention,
)
from src.layers.pos_emb import RoPE
from src.utils.masking_utils import build_intra_doc_attention_mask
from tests.fast.layers.helper import (
    ATTN_IMPLEMENTATION,
    MASK_KIND,
    make_attn_mask,
    skip_if_unsupported,
)
from tests.fast.layers._refs import (
    COMPOUND_DTYPES,
    gqa_ref,
    mha_ref,
    mla_ref,
    sdpa_ref,
)


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
    mask = build_intra_doc_attention_mask(
        pos, q.device, q.dtype, attn_implementation="sdpa"
    )
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


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_mha_attn_mask_output_shape(kind, impl, device):
    skip_if_unsupported(impl, device)
    mha = MultiHeadAttention(
        d_model=64, n_heads=4, dropout=0.0, attn_implementation=impl
    )
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, _ = make_attn_mask(kind, impl, pos, x.dtype)
    assert mha(x, attn_mask=attn_mask).shape == (2, 8, 64)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_mha_intra_doc_mask_blocks_cross_doc_attention(impl, device):
    """Token in doc1 must not be influenced by tokens in doc0. Intra-doc only —
    the causal mask doesn't enforce doc boundaries."""
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mha = MultiHeadAttention(
        d_model=64, n_heads=4, dropout=0.0, attn_implementation=impl
    )
    mha.eval()

    x = torch.randn(1, 4, 64)
    pos = torch.tensor([[0, 1, 0, 1]])  # doc0=[pos0,pos1], doc1=[pos2,pos3]
    attn_mask, _ = make_attn_mask("intra_doc", impl, pos, x.dtype)

    out_base = mha(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64)
    x2[0, 1, :] = torch.randn(64)
    out_modified = mha(x2, attn_mask=attn_mask)

    assert torch.allclose(out_base[0, 2:], out_modified[0, 2:], atol=1e-5)
    assert not torch.allclose(out_base[0, :2], out_modified[0, :2], atol=1e-5)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_mha_causal_mask_blocks_future(impl, device):
    """Modifying the last token must not change earlier-token outputs."""
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mha = MultiHeadAttention(
        d_model=64, n_heads=4, dropout=0.0, attn_implementation=impl
    )
    mha.eval()

    x = torch.randn(1, 8, 64)
    pos = torch.arange(8).unsqueeze(0)
    attn_mask, _ = make_attn_mask("causal", impl, pos, x.dtype)

    out_base = mha(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = mha(x2, attn_mask=attn_mask)
    assert torch.allclose(out_base[0, :7], out_modified[0, :7], atol=1e-5)


# ============================= GroupedQueryAttention behavior =============================


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_gqa_attn_mask_output_shape(kind, impl, device):
    skip_if_unsupported(impl, device)
    gqa = GroupedQueryAttention(
        d_model=64, n_heads=4, n_kv_heads=2, dropout=0.0, attn_implementation=impl
    )
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, _ = make_attn_mask(kind, impl, pos, x.dtype)
    assert gqa(x, attn_mask=attn_mask).shape == (2, 8, 64)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gqa_intra_doc_mask_blocks_cross_doc_attention(impl, device):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(
        d_model=64, n_heads=4, n_kv_heads=2, dropout=0.0, attn_implementation=impl
    )
    gqa.eval()

    x = torch.randn(1, 4, 64)
    pos = torch.tensor([[0, 1, 0, 1]])
    attn_mask, _ = make_attn_mask("intra_doc", impl, pos, x.dtype)

    out_base = gqa(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64)
    x2[0, 1, :] = torch.randn(64)
    out_modified = gqa(x2, attn_mask=attn_mask)

    assert torch.allclose(out_base[0, 2:], out_modified[0, 2:], atol=1e-5)
    assert not torch.allclose(out_base[0, :2], out_modified[0, :2], atol=1e-5)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gqa_causal_mask_blocks_future(impl, device):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(
        d_model=64, n_heads=4, n_kv_heads=2, dropout=0.0, attn_implementation=impl
    )
    gqa.eval()

    x = torch.randn(1, 8, 64)
    pos = torch.arange(8).unsqueeze(0)
    attn_mask, _ = make_attn_mask("causal", impl, pos, x.dtype)

    out_base = gqa(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = gqa(x2, attn_mask=attn_mask)
    assert torch.allclose(out_base[0, :7], out_modified[0, :7], atol=1e-5)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gqa_with_rope_output_shape(impl, device):
    skip_if_unsupported(impl, device)
    rope = RoPE(d_head=16, max_seq_len=32)
    gqa = GroupedQueryAttention(
        d_model=64, n_heads=4, n_kv_heads=2, dropout=0.0, attn_implementation=impl
    )
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, _ = make_attn_mask("causal", impl, pos, x.dtype)
    assert gqa(x, rope, position_ids=pos, attn_mask=attn_mask).shape == (2, 8, 64)


# ============================= MHA / GQA numerical parity vs eager ref =============================


def _mha_ref_call(mha, x, attn_mask=None):
    return mha_ref(
        x,
        mha.q_proj,
        mha.k_proj,
        mha.v_proj,
        mha.o_proj,
        mha.n_heads,
        q_norm=getattr(mha, "q_norm", None),
        k_norm=getattr(mha, "k_norm", None),
        attn_mask=attn_mask,
    )


def _gqa_ref_call(gqa, x, attn_mask=None):
    return gqa_ref(
        x,
        gqa.q_proj,
        gqa.k_proj,
        gqa.v_proj,
        gqa.o_proj,
        gqa.n_heads,
        gqa.n_kv_heads,
        q_norm=getattr(gqa, "q_norm", None),
        k_norm=getattr(gqa, "k_norm", None),
        attn_mask=attn_mask,
    )


MODULE_DTYPES = COMPOUND_DTYPES


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_mha_matches_ref_attn_sink(kind, device, dtype, atol):
    skip_if_unsupported("flex_attention", device)
    torch.manual_seed(17)
    mha = MultiHeadAttention(
        d_model=64,
        n_heads=4,
        qk_norm=True,
        bias=True,
        attn_implementation="flex_attention",
        attn_sink=True,
    ).to(dtype)
    with torch.no_grad():
        mha.sinks.copy_(torch.tensor([-1.25, -0.25, 0.75, 1.75]))
    x = torch.randn(2, 8, 64, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    positions = torch.arange(8).repeat(2, 1)
    if kind == "intra_doc":
        positions[0, 3:] -= 3
        positions[1, 5:] -= 5
    mask, ref_mask = make_attn_mask(
        kind, "flex_attention", positions, dtype, attn_sink=True
    )
    rope = RoPE(d_head=16, max_seq_len=8).to(dtype)
    out = mha(x, rope=rope, position_ids=positions, attn_mask=mask)
    out_ref = mha_ref(
        x_ref,
        mha.q_proj,
        mha.k_proj,
        mha.v_proj,
        mha.o_proj,
        mha.n_heads,
        q_norm=getattr(mha, "q_norm", None),
        k_norm=getattr(mha, "k_norm", None),
        rope=rope,
        position_ids=positions,
        attn_mask=ref_mask,
        sinks=mha.sinks,
    )
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, rtol=0, atol=atol)
    grad_output = torch.randn_like(out)
    parameters = tuple(mha.parameters())
    actual_grads = torch.autograd.grad(out, (x, *parameters), grad_output)
    expected_grads = torch.autograd.grad(out_ref, (x_ref, *parameters), grad_output)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_gqa_matches_ref_attn_sink(kind, device, dtype, atol):
    skip_if_unsupported("flex_attention", device)
    torch.manual_seed(17)
    gqa = GroupedQueryAttention(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        qk_norm=True,
        bias=True,
        attn_implementation="flex_attention",
        attn_sink=True,
    ).to(dtype)
    with torch.no_grad():
        gqa.sinks.copy_(torch.tensor([-1.25, -0.25, 0.75, 1.75]))
    x = torch.randn(2, 8, 64, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    positions = torch.arange(8).repeat(2, 1)
    if kind == "intra_doc":
        positions[0, 3:] -= 3
        positions[1, 5:] -= 5
    mask, ref_mask = make_attn_mask(
        kind, "flex_attention", positions, dtype, attn_sink=True
    )
    rope = RoPE(d_head=16, max_seq_len=8).to(dtype)
    out = gqa(x, rope=rope, position_ids=positions, attn_mask=mask)
    out_ref = gqa_ref(
        x_ref,
        gqa.q_proj,
        gqa.k_proj,
        gqa.v_proj,
        gqa.o_proj,
        gqa.n_heads,
        gqa.n_kv_heads,
        q_norm=getattr(gqa, "q_norm", None),
        k_norm=getattr(gqa, "k_norm", None),
        rope=rope,
        position_ids=positions,
        attn_mask=ref_mask,
        sinks=gqa.sinks,
    )
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, rtol=0, atol=atol)
    grad_output = torch.randn_like(out)
    parameters = tuple(gqa.parameters())
    actual_grads = torch.autograd.grad(out, (x, *parameters), grad_output)
    expected_grads = torch.autograd.grad(out_ref, (x_ref, *parameters), grad_output)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
@pytest.mark.parametrize("kind", MASK_KIND)
@pytest.mark.parametrize("q_lora_rank", [0, 24])
def test_mla_matches_ref_attn_sink(kind, q_lora_rank, device, dtype, atol):
    skip_if_unsupported("flex_attention", device)
    torch.manual_seed(17)
    mla = MultiHeadLatentAttention(
        d_model=64,
        n_heads=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=16,
        v_head_dim=16,
        kv_lora_rank=32,
        q_lora_rank=q_lora_rank,
        bias=True,
        attn_implementation="flex_attention",
        attn_sink=True,
    ).to(dtype)
    with torch.no_grad():
        mla.sinks.copy_(torch.tensor([-1.25, -0.25, 0.75, 1.75]))
    x = torch.randn(2, 8, 64, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    positions = torch.arange(8).repeat(2, 1)
    if kind == "intra_doc":
        positions[0, 3:] -= 3
        positions[1, 5:] -= 5
    mask, ref_mask = make_attn_mask(
        kind, "flex_attention", positions, dtype, attn_sink=True
    )
    rope = RoPE(d_head=16, max_seq_len=8).to(dtype)
    out = mla(x, rope=rope, position_ids=positions, attn_mask=mask)
    out_ref = mla_ref(
        mla,
        x_ref,
        rope=rope,
        position_ids=positions,
        attn_mask=ref_mask,
        sinks=mla.sinks,
    )
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, rtol=0, atol=atol)
    grad_output = torch.randn_like(out)
    parameters = tuple(mla.parameters())
    actual_grads = torch.autograd.grad(out, (x, *parameters), grad_output)
    expected_grads = torch.autograd.grad(out_ref, (x_ref, *parameters), grad_output)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attn_sink_forward_saturation_precision(dtype, device):
    skip_if_unsupported("flex_attention", device)
    mha = MultiHeadAttention(
        d_model=16,
        n_heads=1,
        bias=False,
        attn_implementation="flex_attention",
        attn_sink=True,
    )
    with torch.no_grad():
        mha.q_proj.weight.zero_()
        mha.k_proj.weight.zero_()
        mha.v_proj.weight.copy_(torch.eye(16))
        mha.o_proj.weight.copy_(torch.eye(16))
        mha.sinks.fill_(-17)
    x = torch.ones(1, 1, 16, requires_grad=True)
    positions = torch.zeros(1, 1, dtype=torch.long)
    mask, _ = make_attn_mask(
        "intra_doc", "flex_attention", positions, dtype, attn_sink=True
    )
    attention = torch.compile(mha)
    with torch.autocast("cuda", dtype=dtype):
        out = attention(x, attn_mask=mask)
    actual_grad = torch.autograd.grad(out.float().sum(), mha.sinks)[0]

    sink_logits = mha.sinks.view(1, 1, 1, 1)
    logits = torch.zeros(1, 1, 1, 1)
    probabilities = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)
    expected_out = probabilities[..., :-1] @ x.view(1, 1, 1, 16)
    expected_grad = torch.autograd.grad(expected_out.sum(), mha.sinks)[0]

    torch.testing.assert_close(
        out, expected_out.view_as(out).to(out.dtype), rtol=0, atol=0
    )
    assert out.dtype == dtype
    assert torch.count_nonzero(actual_grad)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=3e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attn_sink_backward_saturation_precision(dtype, device):
    skip_if_unsupported("flex_attention", device)
    mha = MultiHeadAttention(16, 1, attn_sink=True)
    with torch.no_grad():
        for projection in [mha.q_proj, mha.k_proj, mha.v_proj, mha.o_proj]:
            projection.weight.copy_(torch.eye(16))
        mha.sinks.fill_(-5)
    x = torch.ones(1, 128, 16)
    positions = torch.arange(128).unsqueeze(0)
    positions[:, 64:] -= 64
    mask, _ = make_attn_mask(
        "intra_doc", "flex_attention", positions, dtype, attn_sink=True
    )
    attention = torch.compile(mha)
    with torch.autocast("cuda", dtype=dtype):
        output = attention(x, attn_mask=mask)
        # Each document's first token has exactly one visible real key.
        loss = output[positions == 0].float().sum()
    query_grad, key_grad, sink_grad = torch.autograd.grad(
        loss, (mha.q_proj.weight, mha.k_proj.weight, mha.sinks)
    )

    probabilities = torch.tensor([9, 0], dtype=torch.float32).softmax(0)
    real_probability, sink_probability = probabilities.unbind()
    documents = (positions == 0).sum()
    expected_grad = 4 * real_probability * sink_probability * documents
    expected_sink_grad = -4 * expected_grad
    assert output.dtype == dtype
    for grad in [query_grad, key_grad]:
        assert grad.dtype == torch.float32
        assert torch.count_nonzero(grad) == grad.numel()
        torch.testing.assert_close(
            grad, expected_grad.expand_as(grad), rtol=0, atol=5e-5
        )
    torch.testing.assert_close(
        sink_grad,
        expected_sink_grad.expand_as(sink_grad),
        rtol=0,
        atol=2e-7,
    )
    torch.testing.assert_close(
        output[positions == 0],
        real_probability.to(dtype).expand_as(output[positions == 0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("sequence_length", [127, 128, 129])
@pytest.mark.parametrize("kind", MASK_KIND)
def test_multi_head_attention_forward_sink_mask(kind, sequence_length, device):
    skip_if_unsupported("flex_attention", device)
    mha = MultiHeadAttention(
        d_model=16,
        n_heads=1,
        bias=False,
        attn_implementation="flex_attention",
        attn_sink=True,
    )
    with torch.no_grad():
        mha.q_proj.weight.zero_()
        mha.k_proj.weight.zero_()
        mha.v_proj.weight.copy_(torch.eye(16))
        mha.o_proj.weight.copy_(torch.eye(16))
        mha.sinks.zero_()

    x = torch.arange(1, sequence_length + 1, dtype=torch.float32).view(1, -1, 1)
    x = x.expand(-1, -1, 16)
    position_ids = torch.arange(sequence_length).unsqueeze(0)
    if kind == "causal":
        mask, _ = make_attn_mask(
            "causal", "flex_attention", position_ids, x.dtype, attn_sink=True
        )
        attend = torch.ones(sequence_length, sequence_length, dtype=torch.bool).tril()
    else:
        position_ids[:, sequence_length // 2 :] -= sequence_length // 2
        mask, _ = make_attn_mask(
            "intra_doc", "flex_attention", position_ids, x.dtype, attn_sink=True
        )
        doc_offsets = position_ids - torch.arange(sequence_length).unsqueeze(0)
        attend = (doc_offsets.T == doc_offsets).squeeze(0)
        attend &= torch.ones(sequence_length, sequence_length, dtype=torch.bool).tril()

    out = mha(x, attn_mask=mask)
    weights = attend.to(x.dtype) / (attend.sum(dim=-1, keepdim=True) + 1)
    expected = weights.view(1, sequence_length, sequence_length) @ x
    torch.testing.assert_close(out, expected, rtol=0, atol=9e-5)


@pytest.mark.parametrize("compile_attention", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_multi_head_attention_forward_sink_precision(
    compile_attention, use_mask, device
):
    skip_if_unsupported("flex_attention", device)
    mha = MultiHeadAttention(
        d_model=16,
        n_heads=1,
        bias=False,
        attn_implementation="flex_attention",
        attn_sink=True,
    )
    with torch.no_grad():
        mha.q_proj.weight.zero_()
        mha.k_proj.weight.zero_()
        mha.v_proj.weight.copy_(torch.eye(16))
        mha.o_proj.weight.copy_(torch.eye(16))
        mha.sinks.copy_(torch.log(torch.tensor([9.0 / 7.0])))

    x = torch.ones(1, 3, 16)
    x[0, 2].fill_(1.015625)
    position_ids = torch.arange(3).unsqueeze(0)
    mask = None
    if use_mask:
        mask, _ = make_attn_mask(
            "causal", "flex_attention", position_ids, x.dtype, attn_sink=True
        )
    attention = torch.compile(mha) if compile_attention else mha
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = attention(x, attn_mask=mask)

    expected = (x.double().sum(dim=1) / (x.shape[1] + mha.sinks.double().exp())).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out[:, -1], expected, rtol=0, atol=0)


@pytest.mark.parametrize("attn_cls", ["mha", "gqa", "mla"])
def test_attention_forward_sink_raise_error(attn_cls):
    kwargs = {"n_kv_heads": 2} if attn_cls == "gqa" else {}
    if attn_cls == "mla":
        kwargs.update(
            qk_nope_head_dim=8,
            qk_rope_head_dim=16,
            v_head_dim=16,
            kv_lora_rank=32,
        )
    module = ATTN_REGISTRY[attn_cls](
        64, 4, attn_implementation="sdpa", attn_sink=True, **kwargs
    )
    with pytest.raises(ValueError):
        module(torch.randn(1, 8, 64))


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_mha_matches_ref_attn_mask(kind, impl, device, dtype, atol):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mha = MultiHeadAttention(
        d_model=64, n_heads=4, dropout=0.0, attn_implementation=impl
    ).to(dtype)
    mha.eval()
    x = torch.randn(1, 4, 64, dtype=dtype)
    pos = torch.tensor([[0, 1, 0, 1]] if kind == "intra_doc" else [[0, 1, 2, 3]])
    attn_mask, ref_mask = make_attn_mask(kind, impl, pos, x.dtype)
    out = mha(x, attn_mask=attn_mask)
    out_ref = _mha_ref_call(mha, x, attn_mask=ref_mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
@pytest.mark.parametrize("kind", MASK_KIND)
def test_gqa_matches_ref_attn_mask(kind, impl, device, dtype, atol):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(
        d_model=64, n_heads=4, n_kv_heads=2, dropout=0.0, attn_implementation=impl
    ).to(dtype)
    gqa.eval()
    x = torch.randn(1, 4, 64, dtype=dtype)
    pos = torch.tensor([[0, 1, 0, 1]] if kind == "intra_doc" else [[0, 1, 2, 3]])
    attn_mask, ref_mask = make_attn_mask(kind, impl, pos, x.dtype)
    out = gqa(x, attn_mask=attn_mask)
    out_ref = _gqa_ref_call(gqa, x, attn_mask=ref_mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


# --- qk_norm parity (Qwen3-style: RMSNorm Q and K before SDPA) ---


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_mha_qk_norm_matches_ref(impl, device, dtype, atol):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mha = MultiHeadAttention(
        d_model=64, n_heads=4, dropout=0.0, qk_norm=True, attn_implementation=impl
    ).to(dtype)
    mha.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, ref_mask = make_attn_mask("causal", impl, pos, x.dtype)
    out = mha(x, attn_mask=attn_mask)
    out_ref = _mha_ref_call(mha, x, attn_mask=ref_mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", MODULE_DTYPES)
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gqa_qk_norm_matches_ref(impl, device, dtype, atol):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
        qk_norm=True,
        attn_implementation=impl,
    ).to(dtype)
    gqa.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, ref_mask = make_attn_mask("causal", impl, pos, x.dtype)
    out = gqa(x, attn_mask=attn_mask)
    out_ref = _gqa_ref_call(gqa, x, attn_mask=ref_mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


# ============================= MultiHeadLatentAttention behavior =============================


def _make_mla(impl, dtype=None, q_lora_rank=0):
    mla = MultiHeadLatentAttention(
        d_model=64,
        n_heads=4,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        kv_lora_rank=32,
        q_lora_rank=q_lora_rank,
        dropout=0.0,
        attn_implementation=impl,
    )
    return mla.to(dtype) if dtype is not None else mla


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
@pytest.mark.parametrize("kind", MASK_KIND)
@pytest.mark.parametrize("q_lora_rank", [0, 24])
def test_mla_attn_mask_output_shape(kind, impl, q_lora_rank, device):
    skip_if_unsupported(impl, device)
    mla = _make_mla(impl, q_lora_rank=q_lora_rank)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, _ = make_attn_mask(kind, impl, pos, x.dtype)
    assert mla(x, attn_mask=attn_mask).shape == (2, 8, 64)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_mla_intra_doc_mask_blocks_cross_doc_attention(impl, device):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mla = _make_mla(impl)
    mla.eval()

    x = torch.randn(1, 4, 64)
    pos = torch.tensor([[0, 1, 0, 1]])
    attn_mask, _ = make_attn_mask("intra_doc", impl, pos, x.dtype)

    out_base = mla(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 0, :] = torch.randn(64)
    x2[0, 1, :] = torch.randn(64)
    out_modified = mla(x2, attn_mask=attn_mask)

    assert torch.allclose(out_base[0, 2:], out_modified[0, 2:], atol=1e-5)
    assert not torch.allclose(out_base[0, :2], out_modified[0, :2], atol=1e-5)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_mla_causal_mask_blocks_future(impl, device):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mla = _make_mla(impl)
    mla.eval()

    x = torch.randn(1, 8, 64)
    pos = torch.arange(8).unsqueeze(0)
    attn_mask, _ = make_attn_mask("causal", impl, pos, x.dtype)

    out_base = mla(x, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = mla(x2, attn_mask=attn_mask)
    assert torch.allclose(out_base[0, :7], out_modified[0, :7], atol=1e-5)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_mla_with_rope_output_shape(impl, device):
    skip_if_unsupported(impl, device)
    mla = _make_mla(impl)
    rope = RoPE(d_head=mla.qk_rope_head_dim, max_seq_len=32)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, _ = make_attn_mask("causal", impl, pos, x.dtype)
    assert mla(x, rope, position_ids=pos, attn_mask=attn_mask).shape == (2, 8, 64)


# --- MLA numerical parity vs eager mla_ref ---


@pytest.mark.parametrize("dtype,atol", COMPOUND_DTYPES)
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
@pytest.mark.parametrize("kind", MASK_KIND)
@pytest.mark.parametrize("q_lora_rank", [0, 24])
def test_mla_matches_ref_attn_mask(kind, impl, q_lora_rank, device, dtype, atol):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mla = _make_mla(impl, dtype=dtype, q_lora_rank=q_lora_rank)
    mla.eval()
    x = torch.randn(1, 4, 64, dtype=dtype)
    pos = torch.tensor([[0, 1, 0, 1]] if kind == "intra_doc" else [[0, 1, 2, 3]])
    attn_mask, ref_mask = make_attn_mask(kind, impl, pos, x.dtype)
    out = mla(x, attn_mask=attn_mask)
    out_ref = mla_ref(mla, x, attn_mask=ref_mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", COMPOUND_DTYPES)
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_mla_with_rope_matches_ref(impl, device, dtype, atol):
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    mla = _make_mla(impl, dtype=dtype)
    mla.eval()
    rope = RoPE(d_head=mla.qk_rope_head_dim, max_seq_len=32).to(dtype)
    x = torch.randn(2, 8, 64, dtype=dtype)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask, ref_mask = make_attn_mask("causal", impl, pos, x.dtype)
    out = mla(x, rope, position_ids=pos, attn_mask=attn_mask)
    out_ref = mla_ref(mla, x, rope=rope, position_ids=pos, attn_mask=ref_mask)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


def test_attn_registry_keys():
    assert ATTN_REGISTRY["mha"] is MultiHeadAttention
    assert ATTN_REGISTRY["gqa"] is GroupedQueryAttention
    assert ATTN_REGISTRY["mla"] is MultiHeadLatentAttention


def test_mha_compute_flops_value():
    # single int: qkv(24576) + o(8192) + attn_matmul(32768) + qk_norm(0)
    f = MultiHeadAttention.compute_flops(64, 128, n_heads=2, bias=False, qk_norm=False)
    assert f == 24576 + 8192 + 32768


def test_gqa_compute_flops_qk_norm():
    off = GroupedQueryAttention.compute_flops(64, 128, n_heads=2, n_kv_heads=1)
    on = GroupedQueryAttention.compute_flops(
        64, 128, n_heads=2, n_kv_heads=1, qk_norm=True
    )
    assert on - off == 3 * (2 + 1) * 32


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (MultiHeadAttention, dict(n_heads=2)),
        (MultiHeadAttention, dict(n_heads=2, bias=True, qk_norm=True)),
        (GroupedQueryAttention, dict(n_heads=4, n_kv_heads=2)),
        (GroupedQueryAttention, dict(n_heads=4, n_kv_heads=2, bias=True, qk_norm=True)),
        (
            MultiHeadLatentAttention,
            dict(
                n_heads=4,
                qk_nope_head_dim=16,
                qk_rope_head_dim=8,
                v_head_dim=16,
                kv_lora_rank=32,
            ),
        ),
        (
            MultiHeadLatentAttention,
            dict(
                n_heads=4,
                qk_nope_head_dim=16,
                qk_rope_head_dim=8,
                v_head_dim=16,
                kv_lora_rank=32,
                q_lora_rank=24,
                bias=True,
            ),
        ),
    ],
)
def test_compute_parameters_matches_module(cls, kwargs):
    d_model = 64
    module = cls(d_model, **kwargs)
    actual = sum(p.numel() for p in module.parameters())
    assert cls.compute_parameters(d_model, **kwargs) == actual
