import pytest
import torch
from src.model.gpt2 import GPT2Model
from src.model.registry import build_model
from src.utils.config import ModelConfig
from tests.fast.helpers import ATTN_IMPLEMENTATION, make_attn_mask, skip_if_unsupported


def _small_config(attn_implementation: str = "sdpa"):
    return ModelConfig(
        arch="gpt2",
        n_layers=2,
        n_heads=2,
        d_model=64,
        vocab_size=256,
        attn_bias=True,
        mlp_bias=True,
        mlp_activation="gelu",
        mlp_gated=False,
        attn_implementation=attn_implementation,
    )


def _pos(B, S):
    return torch.arange(S).unsqueeze(0).expand(B, S)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gpt2_forward_shape(impl, device):
    skip_if_unsupported(impl, device)
    model = GPT2Model(_small_config(impl), max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    pos = _pos(2, 32)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    logits, _ = model(x, position_ids=pos, attn_mask=attn_mask)
    assert logits.shape == (2, 32, 256)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gpt2_loss(impl, device):
    skip_if_unsupported(impl, device)
    model = GPT2Model(_small_config(impl), max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    pos = _pos(2, 32)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    logits, _ = model(x, position_ids=pos, attn_mask=attn_mask)
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


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gpt2_no_bias(impl, device):
    skip_if_unsupported(impl, device)
    config = ModelConfig(
        arch="gpt2",
        n_layers=2,
        n_heads=2,
        d_model=64,
        vocab_size=256,
        attn_bias=False,
        mlp_bias=False,
        attn_implementation=impl,
    )
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    pos = _pos(2, 32)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    logits, _ = model(x, position_ids=pos, attn_mask=attn_mask)
    assert logits.shape == (2, 32, 256)
    # Verify no linear layers (except lm_head which is already bias=False) have bias
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None, f"{name} should have no bias"


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gpt2_attn_bias_only(impl, device):
    skip_if_unsupported(impl, device)
    config = ModelConfig(
        arch="gpt2",
        n_layers=2,
        n_heads=2,
        d_model=64,
        vocab_size=256,
        attn_bias=True,
        mlp_bias=False,
        attn_implementation=impl,
    )
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    pos = _pos(2, 32)
    attn_mask, _ = make_attn_mask("causal", impl, pos, torch.float32)
    logits, _ = model(x, position_ids=pos, attn_mask=attn_mask)
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


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gpt2_forward_with_position_ids(impl, device):
    skip_if_unsupported(impl, device)
    model = GPT2Model(_small_config(impl), max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    position_ids = torch.arange(32).unsqueeze(0).expand(2, -1)
    attn_mask, _ = make_attn_mask("causal", impl, position_ids, torch.float32)
    logits, _ = model(x, position_ids=position_ids, attn_mask=attn_mask)
    assert logits.shape == (2, 32, 256)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gpt2_position_ids_blocks_cross_doc(impl, device):
    """Intra-doc mask: modifying doc0 tokens must not change doc1 token outputs."""
    skip_if_unsupported(impl, device)
    torch.manual_seed(42)
    model = GPT2Model(_small_config(impl), max_seq_len=128)
    model.eval()

    eot_id = 0
    x = torch.randint(1, 256, (1, 16))
    x[0, 4] = eot_id  # EOT at position 4: doc0=[0..4], doc1=[5..15]
    # position_ids: doc0 → 0,1,2,3,4 ; doc1 → 0,1,2,3,4,5,6,7,8,9,10
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    attn_mask, _ = make_attn_mask("intra_doc", impl, position_ids, torch.float32)

    logits_base, _ = model(x, position_ids=position_ids, attn_mask=attn_mask)

    x2 = x.clone()
    x2[0, :4] = torch.randint(1, 256, (4,))  # change doc0 tokens (not the EOT)
    logits_modified, _ = model(x2, position_ids=position_ids, attn_mask=attn_mask)

    # doc1 positions (5..15) must be unaffected
    assert torch.allclose(logits_base[0, 5:], logits_modified[0, 5:], atol=1e-4), (
        "doc1 logits changed when doc0 input tokens were modified"
    )


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_gpt2_causal_mask_blocks_future(impl, device):
    """Causal mask: modifying the last token must not change earlier-token logits."""
    skip_if_unsupported(impl, device)
    torch.manual_seed(42)
    model = GPT2Model(_small_config(impl), max_seq_len=128)
    model.eval()

    x = torch.randint(1, 256, (1, 16))
    position_ids = torch.arange(16).unsqueeze(0)
    attn_mask, _ = make_attn_mask("causal", impl, position_ids, torch.float32)

    logits_base, _ = model(x, position_ids=position_ids, attn_mask=attn_mask)
    x2 = x.clone()
    x2[0, 15] = (x[0, 15] + 1) % 256
    logits_modified, _ = model(x2, position_ids=position_ids, attn_mask=attn_mask)

    assert torch.allclose(logits_base[0, :15], logits_modified[0, :15], atol=1e-4), (
        "earlier-position logits changed when only the last input token changed"
    )
