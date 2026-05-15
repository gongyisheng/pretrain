import torch

from src.model.qwen3 import Qwen3Model
from src.utils.config import ModelConfig
from src.utils.masking_utils import build_causal_mask


# --- Numerical parity vs HuggingFace Qwen3ForCausalLM ---

def test_qwen3_matches_hf_qwen3_with_copied_weights():
    """Our Qwen3Model produces logits within tolerance of HF Qwen3ForCausalLM.

    HF weights are copied into our model layer-by-layer; vocab_size is set to
    256 (already a multiple of 128) so our padding doesn't kick in.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    vocab_size = 256
    d_model, n_layers, n_heads, n_kv_heads = 64, 2, 4, 2
    intermediate_size, max_seq_len, rope_theta = 128, 32, 10000.0
    d_head = d_model // n_heads

    our_cfg = ModelConfig(
        arch="qwen3",
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_model=d_model,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        rope_theta=rope_theta,
        qk_norm=True,
    )
    ours = Qwen3Model(our_cfg, max_seq_len=max_seq_len)
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
            our_block.ln1.weight.copy_(hf_layer.input_layernorm.weight)
            our_block.attn.q_proj.weight.copy_(hf_layer.self_attn.q_proj.weight)
            our_block.attn.k_proj.weight.copy_(hf_layer.self_attn.k_proj.weight)
            our_block.attn.v_proj.weight.copy_(hf_layer.self_attn.v_proj.weight)
            our_block.attn.o_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            our_block.attn.q_norm.weight.copy_(hf_layer.self_attn.q_norm.weight)
            our_block.attn.k_norm.weight.copy_(hf_layer.self_attn.k_norm.weight)
            our_block.ln2.weight.copy_(hf_layer.post_attention_layernorm.weight)
            # HF stores gate and up as separate projections; ours fuses them into gate_up_proj.
            # Concat along output dim so chunk(2, dim=-1) recovers (gate, up).
            our_block.ffn.gate_up_proj.weight.copy_(
                torch.cat([hf_layer.mlp.gate_proj.weight, hf_layer.mlp.up_proj.weight], dim=0)
            )
            our_block.ffn.down_proj.weight.copy_(hf_layer.mlp.down_proj.weight)
        ours.ln_f.weight.copy_(hf.model.norm.weight)

    B, S = 1, 8
    input_ids = torch.randint(0, vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    our_logits, _ = ours(input_ids, position_ids=position_ids)
    hf_logits = hf(input_ids=input_ids, position_ids=position_ids).logits

    assert our_logits.shape == hf_logits.shape
    assert torch.allclose(our_logits, hf_logits, atol=1e-4)


def _tiny_qwen3_config():
    return ModelConfig(
        arch="qwen3",
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_model=64,
        vocab_size=256,
        rope_theta=10000.0,
        qk_norm=True,
    )


def test_qwen3_forward_with_position_ids_shape():
    model = Qwen3Model(_tiny_qwen3_config(), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    logits, _ = model(x, position_ids=pos)
    assert logits.shape == (2, 8, 256)


def test_qwen3_position_ids_blocks_cross_doc():
    """Modifying doc0 tokens must not change doc1 token outputs."""
    torch.manual_seed(0)
    model = Qwen3Model(_tiny_qwen3_config(), max_seq_len=32)
    model.eval()

    eot_id = 0
    x = torch.randint(1, 256, (1, 8))
    x[0, 3] = eot_id  # doc0=[0..3], doc1=[4..7]
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    attn_mask = build_causal_mask(position_ids, x.device, torch.float32)

    logits_base, _ = model(x, position_ids=position_ids, attn_mask=attn_mask)

    x2 = x.clone()
    x2[0, :3] = torch.randint(1, 256, (3,))
    logits_modified, _ = model(x2, position_ids=position_ids, attn_mask=attn_mask)

    assert torch.allclose(logits_base[0, 4:], logits_modified[0, 4:], atol=1e-4), \
        "doc1 logits changed when doc0 tokens were modified"
    assert not torch.allclose(logits_base[0, :3], logits_modified[0, :3], atol=1e-4), \
        "doc0 logits did NOT change when doc0 tokens were modified (model not propagating inputs?)"
