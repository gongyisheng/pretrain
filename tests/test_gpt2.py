import torch
from src.model.components import build_doc_ids, set_backend
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


def test_gpt2_forward_with_doc_ids():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    doc_ids = build_doc_ids(x, eot_token_id=0)
    logits = model(x, doc_ids=doc_ids)
    assert logits.shape == (2, 32, 256)


def test_gpt2_doc_ids_blocks_cross_doc():
    """Modifying doc0 tokens must not change doc1 token outputs."""
    torch.manual_seed(42)
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    model.eval()

    eot_id = 0
    x = torch.randint(1, 256, (1, 16))
    x[0, 4] = eot_id  # EOT at position 4: doc0=[0..4], doc1=[5..15]
    doc_ids = build_doc_ids(x, eot_token_id=eot_id)

    logits_base = model(x, doc_ids=doc_ids)

    x2 = x.clone()
    x2[0, :4] = torch.randint(1, 256, (4,))  # change doc0 tokens
    logits_modified = model(x2, doc_ids=doc_ids)

    # doc1 positions (5..15) must be unaffected
    assert torch.allclose(logits_base[0, 5:], logits_modified[0, 5:], atol=1e-4), \
        "doc1 logits changed when doc0 input tokens were modified"
