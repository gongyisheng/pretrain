import pytest
import torch
import torch.nn as nn
from src.model.components import BaseTransformerBlock, GeluFFN, MultiHeadAttention, set_backend, MoERouter, SparseMoEBlock
from src.model.qwen3_moe import Qwen3MoEModel
from src.utils.config import ModelConfig


@pytest.fixture(autouse=True)
def backend():
    set_backend("torch")


class _MinimalTransformerBlock(BaseTransformerBlock):
    """Minimal concrete subclass of BaseTransformerBlock used in tests."""

    def __init__(self, d_model: int, n_heads: int, intermediate_size: int, dropout_attn: float = 0.0, dropout_ffn: float = 0.0):
        super().__init__(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout_attn)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = GeluFFN(d_model, intermediate_size, dropout_ffn)

    def attn_sublayer(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return self.attn(self.ln1(x))

    def ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ln2(x))


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


def test_transformer_block_output_shape():
    block = _MinimalTransformerBlock(d_model=64, n_heads=4, intermediate_size=256, dropout_attn=0.0)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == (2, 16, 64)


def test_transformer_block_residual():
    block = _MinimalTransformerBlock(d_model=64, n_heads=4, intermediate_size=256, dropout_attn=0.0)
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
        rope_theta=10000.0,
        qk_norm=True,
        n_experts=4,
        n_experts_per_token=2,
        moe_aux_loss_coef=0.01,
    )


def test_qwen3_moe_model_returns_tuple():
    model = Qwen3MoEModel(_tiny_moe_config(), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 2


def test_qwen3_moe_model_logits_shape():
    model = Qwen3MoEModel(_tiny_moe_config(), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    logits, aux_loss = model(x)
    assert logits.shape == (2, 8, 256)


def test_qwen3_moe_aux_loss_is_scalar_and_nonneg():
    model = Qwen3MoEModel(_tiny_moe_config(), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    _, aux_loss = model(x)
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0.0
