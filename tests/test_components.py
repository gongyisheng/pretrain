import pytest
import torch
import torch.nn as nn
from src.kernel.torch.flashattn import torch_flash_attn
from src.model.components import BaseTransformerBlock, GeluFFN, MultiHeadAttention, set_backend, MoERouter, SparseMoEBlock
from src.model.components import (
    RoPE,
    GroupedQueryAttention,
)
from src.utils.masking_utils import build_causal_mask
from src.model.qwen3 import Qwen3Model
from src.model.qwen3_moe import Qwen3MoEModel
from src.utils.config import ModelConfig


@pytest.fixture(autouse=True)
def backend():
    set_backend("torch")


class _MinimalTransformerBlock(BaseTransformerBlock):
    """Minimal concrete subclass of BaseTransformerBlock used in tests."""

    def __init__(self, d_model: int, n_heads: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = GeluFFN(d_model, intermediate_size, dropout)

    def attn_sublayer(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return self.attn(self.ln1(x))

    def ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ln2(x))


def test_multihead_attention_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(2, 16, 64)
    out = mha(x)
    assert out.shape == (2, 16, 64)


def test_multihead_attention_is_causal():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    mha.eval()
    x = torch.randn(1, 8, 64)
    out_full = mha(x)
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = mha(x2)
    assert torch.allclose(out_full[0, :7], out_modified[0, :7], atol=1e-6)


def test_transformer_block_output_shape():
    block = _MinimalTransformerBlock(d_model=64, n_heads=4, intermediate_size=256, dropout=0.0)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == (2, 16, 64)


def test_transformer_block_residual():
    block = _MinimalTransformerBlock(d_model=64, n_heads=4, intermediate_size=256, dropout=0.0)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert not torch.allclose(out, x)


# --- MoE component tests ---

def test_moe_router_output_shapes():
    router = MoERouter(d_model=64, n_experts=8, n_experts_per_token=2)
    x = torch.randn(4 * 16, 64)  # T=64 tokens
    top_indices, top_weights, router_probs = router(x)
    assert top_indices.shape == (64, 2)
    assert top_weights.shape == (64, 2)
    assert router_probs.shape == (64, 8)


def test_moe_router_indices_in_range():
    router = MoERouter(d_model=64, n_experts=8, n_experts_per_token=2)
    x = torch.randn(32, 64)
    top_indices, _, _ = router(x)
    assert top_indices.min() >= 0
    assert top_indices.max() < 8


def test_moe_router_weights_normalized():
    router = MoERouter(d_model=64, n_experts=8, n_experts_per_token=2, normalize=True)
    x = torch.randn(32, 64)
    _, top_weights, _ = router(x)
    sums = top_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_moe_router_weights_unnormalized():
    router = MoERouter(d_model=64, n_experts=8, n_experts_per_token=2, normalize=False)
    x = torch.randn(32, 64)
    _, top_weights, router_probs = router(x)
    # Weights should be positive (from softmax)
    assert (top_weights > 0).all()
    # Weights should NOT generally sum to 1 (they are raw softmax probabilities)
    sums = top_weights.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_sparse_moe_block_output_shape():
    block = SparseMoEBlock(d_model=64, intermediate_size=128, n_experts=4, n_experts_per_token=2)
    x = torch.randn(2, 8, 64)
    out, aux_loss = block(x)
    assert out.shape == (2, 8, 64)


def test_sparse_moe_block_aux_loss_is_scalar_and_nonneg():
    block = SparseMoEBlock(d_model=64, intermediate_size=128, n_experts=4, n_experts_per_token=2)
    x = torch.randn(2, 8, 64)
    _, aux_loss = block(x)
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0.0


def test_sparse_moe_block_aux_loss_has_grad():
    block = SparseMoEBlock(d_model=64, intermediate_size=128, n_experts=4, n_experts_per_token=2)
    x = torch.randn(2, 8, 64)
    _, aux_loss = block(x)
    aux_loss.backward()
    # router gate should receive gradient
    assert block.router.gate.weight.grad is not None


def _tiny_moe_config():
    return ModelConfig(
        arch="qwen3_moe",
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_model=64,
        intermediate_size=64,
        vocab_size=256,
        dropout=0.0,
        rope_theta=10000.0,
        qk_norm=True,
        n_experts=4,
        n_experts_per_token=2,
        moe_aux_loss_coef=0.01,
    )


def _moe_pos(B, S):
    return torch.arange(S).unsqueeze(0).expand(B, S)


def test_qwen3_moe_model_returns_tuple():
    model = Qwen3MoEModel(_tiny_moe_config(), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    out = model(x, position_ids=_moe_pos(2, 8))
    assert isinstance(out, tuple) and len(out) == 2


def test_qwen3_moe_model_logits_shape():
    model = Qwen3MoEModel(_tiny_moe_config(), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    logits, aux_loss = model(x, position_ids=_moe_pos(2, 8))
    assert logits.shape == (2, 8, 256)


def test_qwen3_moe_aux_loss_is_scalar_and_nonneg():
    model = Qwen3MoEModel(_tiny_moe_config(), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    _, aux_loss = model(x, position_ids=_moe_pos(2, 8))
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0.0



def test_torch_flash_attn_with_attn_mask():
    B, H, S, D = 2, 4, 8, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    mask = torch.zeros(B, 1, S, S)
    mask.masked_fill_(~torch.ones(S, S).tril().bool(), float('-inf'))
    out = torch_flash_attn(q, k, v, attn_mask=mask)
    assert out.shape == (B, H, S, D)


def test_mha_attn_mask_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    attn_mask = build_causal_mask(pos, device=x.device, dtype=x.dtype)
    out = mha(x, attn_mask=attn_mask)
    assert out.shape == (2, 8, 64)


def test_mha_attn_mask_blocks_cross_doc_attention():
    """Token in doc1 must not be influenced by tokens in doc0."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
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


def test_gqa_forward_with_position_ids_shape():
    """GroupedQueryAttention accepts position_ids and returns correct shape."""
    rope = RoPE(d_head=16, max_seq_len=32)
    gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, dropout=0.0)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    out = gqa(x, rope, position_ids=pos)
    assert out.shape == (2, 8, 64)


def _tiny_qwen3_config():
    return ModelConfig(
        arch="qwen3",
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_model=64,
        vocab_size=256,
        dropout=0.0,
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
