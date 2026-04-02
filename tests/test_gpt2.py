import torch
from src.model.components import set_backend
from src.model.gpt2 import GPT2Model
from src.model.registry import build_model
from src.utils.config import ModelConfig
from src.utils.masking_utils import build_causal_mask

set_backend("torch")


def _small_config():
    return ModelConfig(arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=256)


def _pos(B, S):
    return torch.arange(S).unsqueeze(0).expand(B, S)


def test_gpt2_forward_shape():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    logits, _ = model(x, position_ids=_pos(2, 32))
    assert logits.shape == (2, 32, 256)


def test_gpt2_loss():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    logits, _ = model(x, position_ids=_pos(2, 32))
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


def test_gpt2_forward_with_position_ids():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    position_ids = torch.arange(32).unsqueeze(0).expand(2, -1)
    logits, _ = model(x, position_ids=position_ids)
    assert logits.shape == (2, 32, 256)


def test_gpt2_position_ids_blocks_cross_doc():
    """Modifying doc0 tokens must not change doc1 token outputs."""
    torch.manual_seed(42)
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    model.eval()

    eot_id = 0
    x = torch.randint(1, 256, (1, 16))
    x[0, 4] = eot_id  # EOT at position 4: doc0=[0..4], doc1=[5..15]
    # position_ids: doc0 → 0,1,2,3,4 ; doc1 → 0,1,2,3,4,5,6,7,8,9,10
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    attn_mask = build_causal_mask(position_ids, x.device, torch.float32)

    logits_base, _ = model(x, position_ids=position_ids, attn_mask=attn_mask)

    x2 = x.clone()
    x2[0, :4] = torch.randint(1, 256, (4,))   # change doc0 tokens (not the EOT)
    logits_modified, _ = model(x2, position_ids=position_ids, attn_mask=attn_mask)

    # doc1 positions (5..15) must be unaffected
    assert torch.allclose(logits_base[0, 5:], logits_modified[0, 5:], atol=1e-4), \
        "doc1 logits changed when doc0 input tokens were modified"
