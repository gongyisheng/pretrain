import pytest
import torch
from src.model.components import set_backend
from src.model.gpt2 import GPT2Model
from src.model.registry import build_model
from src.utils.config import ModelConfig

set_backend("torch")


def _small_config():
    return ModelConfig(arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=256, attn_bias=True, mlp_bias=True)


def test_gpt2_forward_shape():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 256)


def test_gpt2_loss():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)
    )
    assert loss.item() > 0
    assert loss.requires_grad


def test_gpt2_param_count():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    assert n_params < 1_000_000


def test_gpt2_no_bias():
    config = ModelConfig(arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=256, attn_bias=False, mlp_bias=False)
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 256)
    # Verify no linear layers (except lm_head which is already bias=False) have bias
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None, f"{name} should have no bias"


def test_gpt2_attn_bias_only():
    config = ModelConfig(arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=256, attn_bias=True, mlp_bias=False)
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 256)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "attn" in name and "lm_head" not in name:
                assert module.bias is not None, f"{name} should have bias"
            elif "ffn" in name:
                assert module.bias is None, f"{name} should have no bias"


def test_registry_build_model():
    config = _small_config()

    class FakeTrainConfig:
        model = config
        max_seq_len = 128

    model = build_model(FakeTrainConfig())
    assert isinstance(model, GPT2Model)
