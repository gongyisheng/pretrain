import torch

from src.layers.moe import MoERouter, SparseMoEBlock


# --- Numerical parity vs HF Qwen3MoeSparseMoeBlock ---

def test_sparse_moe_block_matches_hf_qwen3_moe():
    """SparseMoEBlock output matches HF Qwen3MoeSparseMoeBlock with copied weights.

    HF's Qwen3MoeExperts uses the same stacked (E, 2*I, D) / (E, D, I) layout as
    ours, so weight copies are a direct memcpy. Capacity dropping is disabled
    (capacity_factor=None) to match HF's no-drop behavior.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeSparseMoeBlock,
    )

    torch.manual_seed(0)
    d_model, inter, n_experts, top_k = 64, 32, 4, 2
    B, S = 2, 8

    ours = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_experts=n_experts,
        n_experts_per_token=top_k,
        dropout_ffn=0.0,
    )
    # Expert weights are torch.empty by default; initialize before parity check.
    with torch.no_grad():
        torch.nn.init.normal_(ours.expert_gate_up, std=0.02)
        torch.nn.init.normal_(ours.expert_down, std=0.02)
    ours.eval()

    hf_cfg = Qwen3MoeConfig(
        hidden_size=d_model,
        moe_intermediate_size=inter,
        num_experts=n_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
    )
    hf = Qwen3MoeSparseMoeBlock(hf_cfg)
    hf.eval()

    with torch.no_grad():
        hf.gate.weight.copy_(ours.router.gate.weight)
        hf.experts.gate_up_proj.copy_(ours.expert_gate_up)
        hf.experts.down_proj.copy_(ours.expert_down)

    x = torch.randn(B, S, d_model)
    our_out, _ = ours(x)
    hf_out = hf(x)

    assert torch.allclose(our_out, hf_out, atol=1e-5)


# --- MoE router ---

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


# --- SparseMoEBlock ---

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
