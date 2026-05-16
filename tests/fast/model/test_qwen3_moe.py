import pytest
import torch

from src.model.qwen3_moe import Qwen3MoEModel
from src.utils.config import ModelConfig
from tests.fast.helpers import ATTN_IMPLEMENTATION, make_attn_mask, skip_if_unsupported


def _tiny_moe_config(attn_implementation: str = "sdpa"):
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
        moe_n_experts=4,
        moe_n_experts_per_token=2,
        moe_aux_loss_coef=0.01,
        attn_implementation=attn_implementation,
    )


def _moe_pos(B, S):
    return torch.arange(S).unsqueeze(0).expand(B, S)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_qwen3_moe_model_returns_tuple(impl, device):
    skip_if_unsupported(impl, device)
    model = Qwen3MoEModel(_tiny_moe_config(impl), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    pos = _moe_pos(2, 8)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    out = model(x, position_ids=pos, attn_mask=attn_mask)
    assert isinstance(out, tuple) and len(out) == 2


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_qwen3_moe_model_logits_shape(impl, device):
    skip_if_unsupported(impl, device)
    model = Qwen3MoEModel(_tiny_moe_config(impl), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    pos = _moe_pos(2, 8)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    logits, aux_loss = model(x, position_ids=pos, attn_mask=attn_mask)
    assert logits.shape == (2, 8, 256)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_qwen3_moe_aux_loss_is_scalar_and_nonneg(impl, device):
    skip_if_unsupported(impl, device)
    model = Qwen3MoEModel(_tiny_moe_config(impl), max_seq_len=32)
    x = torch.randint(0, 256, (2, 8))
    pos = _moe_pos(2, 8)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    _, aux_loss = model(x, position_ids=pos, attn_mask=attn_mask)
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0.0
