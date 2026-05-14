"""Tests for MetricsTracker.compute_layer_grad_norms."""

import pytest
import torch

from src.model.components import set_backend
from src.model.registry import build_model
from src.training.metrics import MetricsTracker
from src.utils.config import ModelConfig

set_backend("torch")

# ---------------------------------------------------------------------------
# Small model configs matching real architectures
# ---------------------------------------------------------------------------

_GPT2_CFG = ModelConfig(
    arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=256
)
_QWEN3_CFG = ModelConfig(
    arch="qwen3", n_layers=2, n_heads=2, n_kv_heads=1, d_model=64, vocab_size=256, qk_norm=True,
)
_QWEN3_MOE_CFG = ModelConfig(
    arch="qwen3_moe", n_layers=2, n_heads=2, n_kv_heads=1, d_model=64, vocab_size=256,
    qk_norm=True, moe_n_experts=4, moe_n_experts_per_token=2, moe_intermediate_size=128,
)
_GPT2_ATTN_RES_CFG = ModelConfig(
    arch="gpt2", n_layers=4, n_heads=2, d_model=64, vocab_size=256,
    attn_res=True, attn_res_block_size=2,
)
_QWEN3_ATTN_RES_CFG = ModelConfig(
    arch="qwen3", n_layers=4, n_heads=2, n_kv_heads=1, d_model=64, vocab_size=256,
    qk_norm=True, attn_res=True, attn_res_block_size=2,
)


class _FakeTrainConfig:
    """Minimal object accepted by build_model (needs .model and .max_seq_len)."""

    def __init__(self, model_cfg: ModelConfig):
        self.model = model_cfg
        self.max_seq_len = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate_grads(model, vocab_size: int):
    """Run a forward/backward pass to populate .grad on all parameters."""
    idx = torch.randint(0, vocab_size, (1, 16))
    out = model(idx)
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


_ALL_CFGS = [_GPT2_CFG, _QWEN3_CFG, _QWEN3_MOE_CFG, _GPT2_ATTN_RES_CFG, _QWEN3_ATTN_RES_CFG]
_ALL_IDS = ["gpt2", "qwen3", "qwen3_moe", "gpt2_attn_res", "qwen3_attn_res"]


@pytest.mark.parametrize("model_cfg", _ALL_CFGS, ids=_ALL_IDS)
def test_layer_grad_norms_plain_model(model_cfg):
    """Per-module grad norms should have one key per module on a plain model."""
    model = build_model(_FakeTrainConfig(model_cfg))
    _populate_grads(model, model_cfg.vocab_size)
    result = MetricsTracker.compute_layer_grad_norms(model)
    _assert_keys_match(result, model)


@pytest.mark.parametrize("model_cfg", _ALL_CFGS, ids=_ALL_IDS)
def test_layer_grad_norms_compiled_model(model_cfg):
    """Per-module grad norms must work after torch.compile (which prepends _orig_mod.)."""
    model = build_model(_FakeTrainConfig(model_cfg))
    compiled = torch.compile(model, backend="eager")
    _populate_grads(compiled, model_cfg.vocab_size)
    result = MetricsTracker.compute_layer_grad_norms(compiled)
    _assert_keys_match(result, compiled)
