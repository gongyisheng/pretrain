import pytest
import torch
import torch.nn as nn

from src.quant.convert import apply_quantization
from src.quant.linear import QuantLinear
from src.utils.config import ModelConfig, TrainConfig, TrainingConfig


def _fp8_capable():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="fp8 needs SM >= 8.9")


class _Tiny(nn.Module):
    def __init__(self, tie=False):
        super().__init__()
        self.token_emb = nn.Embedding(64, 32)
        self.attn = nn.ModuleDict({"q_proj": nn.Linear(32, 32, bias=False)})
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(32, 32, bias=False)})
        self.lm_head = nn.Linear(32, 64, bias=False)
        if tie:
            self.lm_head.weight = self.token_emb.weight


def _cfg(quant, mixed_precision="bf16"):
    return TrainConfig(
        model=ModelConfig(
            d_model=32,
            n_layers=1,
            vocab_size=64,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 2}}],
        ),
        training=TrainingConfig(mixed_precision=mixed_precision, quant=quant),
    )


def test_disabled_is_noop():
    m = _Tiny()
    apply_quantization(m, _cfg({"enabled": False}))
    assert isinstance(m.attn["q_proj"], nn.Linear)
    assert not isinstance(m.attn["q_proj"], QuantLinear)


@pytest.mark.skipif(_fp8_capable(), reason="testing the unsupported-hardware branch")
def test_raises_without_capable_gpu():
    m = _Tiny()
    with pytest.raises(RuntimeError, match="compute capability"):
        apply_quantization(m, _cfg({"enabled": True, "dtype_recipe": "fp8"}))


@fp8_only
def test_swaps_and_excludes_lm_head():
    m = _Tiny().cuda().to(torch.bfloat16)
    apply_quantization(m, _cfg({"enabled": True, "dtype_recipe": "fp8"}))
    assert isinstance(m.attn["q_proj"], QuantLinear)
    assert isinstance(m.mlp["down_proj"], QuantLinear)
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, QuantLinear)
    assert isinstance(m.token_emb, nn.Embedding)


@fp8_only
def test_tie_guard_skips_tied_lm_head():
    m = _Tiny(tie=True).cuda().to(torch.bfloat16)
    # exclude nothing: tie guard alone must keep lm_head unswapped
    cfg = _cfg({"enabled": True, "dtype_recipe": "fp8", "exclude": []})
    with pytest.warns(UserWarning, match="tied"):
        apply_quantization(m, cfg)
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, QuantLinear)
    assert isinstance(m.attn["q_proj"], QuantLinear)  # non-tied still swapped


@fp8_only
def test_include_restricts_scope():
    m = _Tiny().cuda().to(torch.bfloat16)
    cfg = _cfg(
        {"enabled": True, "dtype_recipe": "fp8", "include": ["*attn*"], "exclude": []}
    )
    apply_quantization(m, cfg)
    assert isinstance(m.attn["q_proj"], QuantLinear)
    assert isinstance(m.mlp["down_proj"], nn.Linear)  # not in include


@fp8_only
def test_list_of_rules_first_match_wins():
    m = _Tiny().cuda().to(torch.bfloat16)
    cfg = _cfg(
        [
            {
                "enabled": True,
                "dtype_recipe": "fp8",
                "include": ["*attn*"],
                "exclude": [],
            },
            {
                "enabled": True,
                "dtype_recipe": "fp8",
                "include": ["*mlp*"],
                "exclude": [],
            },
        ]
    )
    apply_quantization(m, cfg)
    assert isinstance(m.attn["q_proj"], QuantLinear)
    assert isinstance(m.mlp["down_proj"], QuantLinear)
