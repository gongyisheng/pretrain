import torch

from src.model.qwen3_moe import Qwen3MoEModel
from src.utils.config import ModelConfig


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
        moe_n_experts=4,
        moe_n_experts_per_token=2,
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
