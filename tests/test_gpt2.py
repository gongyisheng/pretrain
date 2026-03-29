import pytest
import torch
from src.model.components import set_backend
from src.model.gpt2 import GPT2Model
from src.model.registry import build_model
from src.utils.config import ModelConfig

set_backend("torch")


def _small_config():
    return ModelConfig(arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=256, dropout=0.0)


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


def test_registry_build_model():
    config = _small_config()

    class FakeTrainConfig:
        model = config
        max_seq_len = 128

    model = build_model(FakeTrainConfig())
    assert isinstance(model, GPT2Model)
