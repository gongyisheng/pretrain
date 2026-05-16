"""Tests for MetricsTracker.compute_layer_grad_norms."""

import pytest
import torch

from src.model.registry import build_model
from src.training.metrics import MetricsTracker
from src.utils.config import ModelConfig
from tests.fast._attn_helpers import IMPL, make_attn_mask, skip_if_unsupported

# ---------------------------------------------------------------------------
# Small model configs matching real architectures. Factories rather than
# constants so they can pick up the parametrized attn_implementation.
# ---------------------------------------------------------------------------

def _gpt2_cfg(impl):
    return ModelConfig(
        arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=256,
        attn_implementation=impl,
    )


def _qwen3_cfg(impl):
    return ModelConfig(
        arch="qwen3", n_layers=2, n_heads=2, n_kv_heads=1, d_model=64, vocab_size=256, qk_norm=True,
        attn_implementation=impl,
    )


def _qwen3_moe_cfg(impl):
    return ModelConfig(
        arch="qwen3_moe", n_layers=2, n_heads=2, n_kv_heads=1, d_model=64, vocab_size=256,
        qk_norm=True, moe_n_experts=4, moe_n_experts_per_token=2, moe_intermediate_size=128,
        attn_implementation=impl,
    )


def _gpt2_attn_res_cfg(impl):
    return ModelConfig(
        arch="gpt2", n_layers=4, n_heads=2, d_model=64, vocab_size=256,
        residual_cls="attn_res", residual_kwargs={"seal_block_size": 2},
        attn_implementation=impl,
    )


def _qwen3_attn_res_cfg(impl):
    return ModelConfig(
        arch="qwen3", n_layers=4, n_heads=2, n_kv_heads=1, d_model=64, vocab_size=256,
        qk_norm=True, residual_cls="attn_res", residual_kwargs={"seal_block_size": 2},
        attn_implementation=impl,
    )


_CFG_FACTORIES = {
    "gpt2": _gpt2_cfg,
    "qwen3": _qwen3_cfg,
    "qwen3_moe": _qwen3_moe_cfg,
    "gpt2_attn_res": _gpt2_attn_res_cfg,
    "qwen3_attn_res": _qwen3_attn_res_cfg,
}


class _FakeTrainConfig:
    """Minimal object accepted by build_model (needs .model and .max_seq_len)."""

    def __init__(self, model_cfg: ModelConfig):
        self.model = model_cfg
        self.max_seq_len = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate_grads(model, vocab_size: int, impl: str):
    """Run a forward/backward pass to populate .grad on all parameters."""
    idx = torch.randint(0, vocab_size, (1, 16))
    position_ids = torch.arange(16).unsqueeze(0)
    attn_mask, _ = make_attn_mask("causal", impl, position_ids, torch.float32)
    out = model(idx, position_ids=position_ids, attn_mask=attn_mask)
    # MoE models return (logits, aux_loss)
    logits = out[0] if isinstance(out, tuple) else out
    logits.sum().backward()


def _expected_keys(model: torch.nn.Module) -> set[str]:
    """Derive expected grad_norm keys from model's named_parameters."""
    keys = set()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        name = name.removeprefix("_orig_mod.")
        keys.add(f"grad_norm/{name}")
    return keys


def _assert_keys_match(result: dict, model: torch.nn.Module):
    """Check that result keys exactly match expected per-module keys."""
    expected = _expected_keys(model)
    actual = set(result.keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing keys: {missing}"
    assert not extra, f"Unexpected keys: {extra}"
    # All values should be non-negative finite floats
    for k, v in result.items():
        assert v >= 0, f"{k} has negative grad norm: {v}"
        assert torch.isfinite(torch.tensor(v)), f"{k} is not finite: {v}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch_id", list(_CFG_FACTORIES))
@pytest.mark.parametrize("impl", IMPL)
def test_layer_grad_norms_plain_model(arch_id, impl, device):
    """Per-module grad norms should have one key per module on a plain model."""
    skip_if_unsupported(impl, device)
    model_cfg = _CFG_FACTORIES[arch_id](impl)
    model = build_model(_FakeTrainConfig(model_cfg))
    _populate_grads(model, model_cfg.vocab_size, impl)
    result = MetricsTracker.compute_layer_grad_norms(model)
    _assert_keys_match(result, model)


@pytest.mark.parametrize("arch_id", list(_CFG_FACTORIES))
@pytest.mark.parametrize("impl", IMPL)
def test_layer_grad_norms_compiled_model(arch_id, impl, device):
    """Per-module grad norms must work after torch.compile (which prepends _orig_mod.)."""
    skip_if_unsupported(impl, device)
    model_cfg = _CFG_FACTORIES[arch_id](impl)
    model = build_model(_FakeTrainConfig(model_cfg))
    compiled = torch.compile(model, backend="eager")
    _populate_grads(compiled, model_cfg.vocab_size, impl)
    result = MetricsTracker.compute_layer_grad_norms(compiled)
    _assert_keys_match(result, compiled)
