import pytest
import torch

from src.layers.attention import GroupedQueryAttention
from src.layers.mlp import DenseMLPBlock
from src.model.registry import build_model
from src.model.transformer import TransformerLM
from src.utils.config import ModelConfig, TrainConfig
from tests.fast.helpers import ATTN_IMPLEMENTATION, make_attn_mask, skip_if_unsupported


def _cfg(max_seq_len=32, **model_over):
    base = dict(
        d_model=64,
        n_layers=2,
        vocab_size=256,
        attn_cls="gqa",
        attn_kwargs={"n_heads": 4, "n_kv_heads": 2},
        mlp_cls="dense",
        mlp_kwargs={"activation": "silu", "gated": True},
        norm_cls="rmsnorm",
        pos_emb_cls="rope",
        pos_emb_kwargs={"rope_theta": 1e4},
    )
    base.update(model_over)
    return TrainConfig(max_seq_len=max_seq_len, model=ModelConfig(**base))


def test_compute_flops_sums_components():
    cfg = (
        _cfg()
    )  # gqa(n_heads=4,n_kv=2) + dense gated, d_model=64, L=2, T=32, vocab=256
    m = cfg.model
    attn = GroupedQueryAttention.compute_flops(64, 32, **m.attn_kwargs)
    mlp = DenseMLPBlock.compute_flops(
        64, **m.mlp_kwargs
    )  # mlp flops are seq-len-independent
    per_layer = attn + mlp + 2 * 3 * 64  # + two RMSNorms
    expected = 2 * per_layer + 3 * 64 + 2 * 64 * 256  # final norm + lm_head
    assert TransformerLM.compute_flops(m, cfg.max_seq_len) == expected


def _gpt2_cfg(impl):
    return _cfg(
        attn_cls="mha",
        attn_kwargs={
            "n_heads": 2,
            "qk_norm": False,
            "bias": True,
            "attn_implementation": impl,
        },
        mlp_cls="dense",
        mlp_kwargs={"activation": "gelu", "gated": False, "bias": True},
        norm_cls="layernorm",
        pos_emb_cls="learned",
        pos_emb_kwargs={},
    )


def _qwen3_cfg(impl):
    return _cfg(
        attn_cls="gqa",
        attn_kwargs={
            "n_heads": 4,
            "n_kv_heads": 2,
            "qk_norm": True,
            "attn_implementation": impl,
        },
        mlp_cls="dense",
        mlp_kwargs={"activation": "silu", "gated": True},
        norm_cls="rmsnorm",
        pos_emb_cls="rope",
        pos_emb_kwargs={"rope_theta": 1e4},
    )


def _moe_cfg(impl):
    return _cfg(
        attn_cls="gqa",
        attn_kwargs={
            "n_heads": 4,
            "n_kv_heads": 2,
            "qk_norm": True,
            "attn_implementation": impl,
        },
        mlp_cls="moe",
        mlp_kwargs={
            "intermediate_size": 64,
            "n_experts": 4,
            "n_experts_per_token": 2,
            "activation": "silu",
            "gated": True,
        },
        norm_cls="rmsnorm",
        pos_emb_cls="rope",
        pos_emb_kwargs={"rope_theta": 1e4},
    )


def _pos(B, S):
    return torch.arange(S).unsqueeze(0).expand(B, S)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gpt2_style_forward_shape_and_no_aux(impl, device):
    skip_if_unsupported(impl, device)
    model = build_model(_gpt2_cfg(impl))
    x = torch.randint(0, 256, (2, 16))
    pos = _pos(2, 16)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    logits, aux = model(x, position_ids=pos, attn_mask=attn_mask)
    assert logits.shape == (2, 16, model.padded_vocab_size)
    assert aux is None


def test_gpt2_style_weight_tying():
    model = build_model(_gpt2_cfg("sdpa"))
    assert model.lm_head.weight is model.token_emb.weight


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_qwen3_style_forward_no_aux(impl, device):
    skip_if_unsupported(impl, device)
    model = build_model(_qwen3_cfg(impl))
    x = torch.randint(0, 256, (2, 8))
    pos = _pos(2, 8)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    logits, aux = model(x, position_ids=pos, attn_mask=attn_mask)
    assert logits.shape == (2, 8, model.padded_vocab_size)
    assert aux is None


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_moe_forward_returns_aux(impl, device):
    skip_if_unsupported(impl, device)
    model = build_model(_moe_cfg(impl))
    assert model.is_moe
    x = torch.randint(0, 256, (2, 8))
    pos = _pos(2, 8)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    logits, aux = model(x, position_ids=pos, attn_mask=attn_mask)
    assert logits.shape == (2, 8, model.padded_vocab_size)
    assert aux is not None
    assert aux.ndim == 0
    assert aux.item() >= 0.0


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_intra_doc_blocks_cross_doc(impl, device):
    """Intra-doc mask: modifying doc0 tokens must not change doc1 outputs."""
    skip_if_unsupported(impl, device)
    torch.manual_seed(0)
    model = build_model(_qwen3_cfg(impl))
    model.eval()

    x = torch.randint(1, 256, (1, 8))
    x[0, 3] = 0  # eot: doc0=[0..3], doc1=[4..7]
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    attn_mask, _ = make_attn_mask("intra_doc", impl, position_ids, torch.float32)

    base, _ = model(x, position_ids=position_ids, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, :3] = torch.randint(1, 256, (3,))
    modified, _ = model(x2, position_ids=position_ids, attn_mask=attn_mask)

    assert torch.allclose(base[0, 4:], modified[0, 4:], atol=1e-4)
    assert not torch.allclose(base[0, :3], modified[0, :3], atol=1e-4)


def test_transformer_matches_hf_qwen3_with_copied_weights():
    """Unified TransformerLM (Qwen3 config) matches HF Qwen3ForCausalLM logits.

    HF weights are copied layer-by-layer; vocab_size=256 (multiple of 128) so
    our padding doesn't kick in.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    vocab_size = 256
    d_model, n_layers, n_heads, n_kv_heads = 64, 2, 4, 2
    intermediate_size, max_seq_len, rope_theta = 128, 32, 10000.0
    d_head = d_model // n_heads

    our_cfg = ModelConfig(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        attn_cls="gqa",
        attn_kwargs={
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "qk_norm": True,
            "attn_implementation": "sdpa",  # CPU test; flex requires CUDA
        },
        mlp_cls="dense",
        mlp_kwargs={
            "intermediate_size": intermediate_size,
            "activation": "silu",
            "gated": True,
        },
        norm_cls="rmsnorm",
        pos_emb_cls="rope",
        pos_emb_kwargs={"rope_theta": rope_theta},
    )
    ours = TransformerLM(our_cfg, max_seq_len=max_seq_len)
    ours.eval()

    hf_cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=d_head,
        max_position_embeddings=max_seq_len,
        rope_theta=rope_theta,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    hf = Qwen3ForCausalLM(hf_cfg)
    hf.eval()

    with torch.no_grad():
        ours.token_emb.weight.copy_(hf.model.embed_tokens.weight)
        for i in range(n_layers):
            hf_layer = hf.model.layers[i]
            our_block = ours.blocks[i]
            our_block.norm1.weight.copy_(hf_layer.input_layernorm.weight)
            our_block.attn.q_proj.weight.copy_(hf_layer.self_attn.q_proj.weight)
            our_block.attn.k_proj.weight.copy_(hf_layer.self_attn.k_proj.weight)
            our_block.attn.v_proj.weight.copy_(hf_layer.self_attn.v_proj.weight)
            our_block.attn.o_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            our_block.attn.q_norm.weight.copy_(hf_layer.self_attn.q_norm.weight)
            our_block.attn.k_norm.weight.copy_(hf_layer.self_attn.k_norm.weight)
            our_block.norm2.weight.copy_(hf_layer.post_attention_layernorm.weight)
            # HF keeps gate/up separate; ours fuses them into gate_up_proj.
            our_block.mlp.gate_up_proj.weight.copy_(
                torch.cat(
                    [hf_layer.mlp.gate_proj.weight, hf_layer.mlp.up_proj.weight], dim=0
                )
            )
            our_block.mlp.down_proj.weight.copy_(hf_layer.mlp.down_proj.weight)
        ours.ln_f.weight.copy_(hf.model.norm.weight)

    B, S = 1, 8
    input_ids = torch.randint(0, vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    our_logits, _ = ours(input_ids, position_ids=position_ids)
    hf_logits = hf(input_ids=input_ids, position_ids=position_ids).logits

    assert our_logits.shape == hf_logits.shape
    assert torch.allclose(our_logits, hf_logits, atol=1e-4)
