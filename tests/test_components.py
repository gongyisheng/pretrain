import pytest
import torch
import torch.nn as nn
from src.model.components import BaseTransformerBlock, GeluFFN, MultiHeadAttention, set_backend, MoERouter, SparseMoEBlock
from src.model.components import build_doc_ids, build_doc_causal_mask
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


EOT = 0  # <|endoftext|> token ID used in tests


def test_build_doc_ids_single_doc():
    x = torch.tensor([[1, 2, 3, 4, 5]])  # (1, 5)
    ids = build_doc_ids(x, EOT)
    assert ids.shape == (1, 5)
    assert ids.tolist() == [[0, 0, 0, 0, 0]]


def test_build_doc_ids_eot_belongs_to_its_doc():
    # EOT at position 3 → positions 0-3 are doc 0, positions 4+ are doc 1
    x = torch.tensor([[1, 2, 3, EOT, 5, 6]])  # (1, 6)
    ids = build_doc_ids(x, EOT)
    assert ids.tolist() == [[0, 0, 0, 0, 1, 1]]


def test_build_doc_ids_multiple_docs():
    x = torch.tensor([[1, EOT, 2, EOT, 3]])  # (1, 5)
    ids = build_doc_ids(x, EOT)
    assert ids.tolist() == [[0, 0, 1, 1, 2]]


def test_build_doc_ids_batch():
    x = torch.tensor([
        [1, EOT, 2],
        [3, 4, 5],
    ])  # (2, 3)
    ids = build_doc_ids(x, EOT)
    assert ids.tolist() == [[0, 0, 1], [0, 0, 0]]


def test_build_doc_causal_mask_shape():
    x = torch.tensor([[1, EOT, 2, 3]])  # (1, 4)
    ids = build_doc_ids(x, EOT)
    mask = build_doc_causal_mask(ids, device=ids.device, dtype=torch.float32)
    assert mask.shape == (1, 1, 4, 4)


def test_build_doc_causal_mask_blocks_future():
    x = torch.tensor([[1, 2, 3, 4]])
    ids = build_doc_ids(x, EOT)
    mask = build_doc_causal_mask(ids, device=ids.device, dtype=torch.float32)
    m = mask[0, 0]  # (4, 4)
    for i in range(4):
        for j in range(i + 1, 4):
            assert m[i, j].item() == float('-inf'), f"Expected -inf at ({i},{j})"


def test_build_doc_causal_mask_blocks_cross_doc():
    x = torch.tensor([[1, EOT, 2, 3]])  # doc0=[0,1], doc1=[2,3]
    ids = build_doc_ids(x, EOT)
    mask = build_doc_causal_mask(ids, device=ids.device, dtype=torch.float32)
    m = mask[0, 0]  # (4, 4)
    assert m[2, 0].item() == float('-inf')
    assert m[2, 1].item() == float('-inf')
    assert m[2, 2].item() == 0.0


def test_build_doc_causal_mask_allows_same_doc_causal():
    x = torch.tensor([[1, 2, 3, 4]])  # single doc
    ids = build_doc_ids(x, EOT)
    mask = build_doc_causal_mask(ids, device=ids.device, dtype=torch.float32)
    m = mask[0, 0]  # (4, 4)
    for i in range(4):
        for j in range(i + 1):
            assert m[i, j].item() == 0.0, f"Expected 0.0 at ({i},{j})"


from src.kernel.torch.flashattn import torch_flash_attn


def test_torch_flash_attn_with_attn_mask():
    B, H, S, D = 2, 4, 8, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    mask = torch.zeros(B, 1, S, S)
    mask.masked_fill_(~torch.ones(S, S).tril().bool(), float('-inf'))
    out = torch_flash_attn(q, k, v, causal=False, attn_mask=mask)
    assert out.shape == (B, H, S, D)
