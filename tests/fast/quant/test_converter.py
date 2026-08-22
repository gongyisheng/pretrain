import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.layers.mlp import SparseMoEBlock
from src.quant.constants import _INT8_FORMATS
from src.quant.convert import apply_quantization
from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock
from src.utils.config import (
    ModelConfig,
    QuantizationConfig,
    TrainConfig,
    TrainingConfig,
)

INT_FORMATS = sorted(_INT8_FORMATS)


def _spec(recipe="fp8", scale=None, **overrides):
    """A fresh quantization rule dict. Never share one: QuantizationConfig resolves
    recipes by popping them out of the nested dict it is handed."""
    spec = {"enabled": True, "dtype": {"recipe": recipe}, **overrides}
    if scale is not None:
        spec["scale"] = {"recipe": scale}
    return spec


class _Dense(nn.Module):
    def __init__(self, tie=False):
        super().__init__()
        self.token_emb = nn.Embedding(64, 32)
        self.attn = nn.ModuleDict({"q_proj": nn.Linear(32, 32, bias=False)})
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(32, 32, bias=False)})
        self.lm_head = nn.Linear(32, 64, bias=False)
        if tie:
            self.lm_head.weight = self.token_emb.weight


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(64, 32)
        self.mlp = SparseMoEBlock(
            d_model=32,
            intermediate_size=48,
            n_routed_experts=4,
            n_routed_experts_per_token=2,
        )
        self.lm_head = nn.Linear(32, 64, bias=False)


def _cfg(quantization, n_layers=1):
    # QuantizationConfig resolves recipes by popping them out of the dict it is
    # handed, so a case table entry must not be the dict it consumes.
    quantization = copy.deepcopy(quantization)
    return TrainConfig(
        model=ModelConfig(
            d_model=32,
            n_layers=n_layers,
            vocab_size=64,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 2}}],
        ),
        training=TrainingConfig(mixed_precision="bf16", quantization=quantization),
    )


APPLY_CASES = [
    # (quantization, swapped fqns, untouched fqns)
    ({"enabled": False}, (), ("attn.q_proj", "mlp.down_proj")),
    # fp8 and mxfp8 both swap with no hardware preflight, and lm_head is excluded
    # by default while the embedding is never a candidate at all.
    (_spec("fp8"), ("attn.q_proj", "mlp.down_proj"), ("lm_head", "token_emb")),
    (_spec("mxfp8"), ("attn.q_proj", "mlp.down_proj"), ("lm_head", "token_emb")),
    # an include allowlist restricts the scope
    (
        _spec(include=["*attn*"], exclude=[]),
        ("attn.q_proj",),
        ("mlp.down_proj",),
    ),
    # a list of rules: each fqn takes the first rule that claims it
    (
        [
            _spec(include=["*attn*"], exclude=[]),
            _spec(include=["*mlp*"], exclude=[]),
        ],
        ("attn.q_proj", "mlp.down_proj"),
        ("lm_head",),
    ),
    # the int8 series is weight-only but still swaps the module
    *[(_spec(fmt), ("attn.q_proj",), ("lm_head",)) for fmt in INT_FORMATS],
]


@pytest.mark.parametrize("quantization,swapped,untouched", APPLY_CASES)
def test_apply_quantization(quantization, swapped, untouched):
    model = _Dense()
    apply_quantization(model, _cfg(quantization))
    for fqn in swapped:
        assert isinstance(model.get_submodule(fqn), QuantizedLinear), fqn
    for fqn in untouched:
        assert not isinstance(model.get_submodule(fqn), QuantizedLinear), fqn


def test_apply_quantization_skips_a_tied_lm_head(capsys):
    # exclude nothing: the tie guard alone must keep lm_head unswapped, loudly
    model = _Dense(tie=True)
    apply_quantization(model, _cfg(_spec(exclude=[])))
    assert "tied" in capsys.readouterr().out
    assert not isinstance(model.lm_head, QuantizedLinear)
    assert isinstance(model.attn["q_proj"], QuantizedLinear)  # non-tied still swapped


MOE_CASES = [
    (_spec(scale="rowwise"), QuantizedSparseMoEBlock, False),
    (_spec(scale="rowwise", exclude=["mlp"]), SparseMoEBlock, True),
]


@pytest.mark.parametrize("quantization,seam_owner,gate_swapped", MOE_CASES)
def test_apply_quantization_moe(quantization, seam_owner, gate_swapped):
    model = _MoE()
    apply_quantization(model, _cfg(quantization))
    assert isinstance(model.mlp.expert_gate_up, torch.nn.Parameter)
    assert model.mlp.expert_mm.__func__ is seam_owner.expert_mm
    assert isinstance(model.mlp.router.gate, QuantizedLinear) is gate_swapped


def test_apply_quantization_attaches_no_metric_state():
    model = _Dense()
    apply_quantization(model, _cfg(_spec("fp8")))
    swapped = [m for m in model.modules() if isinstance(m, QuantizedLinear)]
    assert swapped and all(not hasattr(m, "quantization_probe") for m in swapped)


def test_apply_quantization_accepts_a_bare_config_namespace():
    # the converter only reads config.training.quantization, so a plain namespace of
    # already-built rules is enough — nothing else on TrainConfig is consulted
    model = nn.Sequential(nn.Linear(64, 64))
    rule = QuantizationConfig(enabled=True, dtype={"recipe": "int8"}, include=["0"])
    config = SimpleNamespace(training=SimpleNamespace(quantization=[rule]))
    apply_quantization(model, config)
    assert isinstance(model[0], QuantizedLinear)


class _Block(nn.Module):
    """One layer of a stacked model: the attn/mlp split TransformerLM quantizes."""

    def __init__(self):
        super().__init__()
        self.attn = nn.ModuleDict({"q_proj": nn.Linear(32, 32, bias=False)})
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(32, 32, bias=False)})


class _Stacked(nn.Module):
    """Mini TransformerLM: per-layer blocks under `blocks`, standalone lm_head."""

    def __init__(self, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_layers)])
        self.lm_head = nn.Linear(32, 64, bias=False)


def _blocks_flattened(model):
    return [
        quantized
        for block in model.blocks
        for quantized in (block.attn["q_proj"], block.mlp["down_proj"])
    ]


def test_apply_quantization_layer_idx_restricts_layers():
    model = _Stacked(n_layers=4)
    apply_quantization(model, _cfg(_spec("fp8", layer_idx=[0, 2]), n_layers=4))
    got = _blocks_flattened(model)
    assert [isinstance(m, QuantizedLinear) for m in got] == [
        True,
        True,  # block 0
        False,
        False,  # block 1
        True,
        True,  # block 2
        False,
        False,  # block 3
    ]
    assert not isinstance(model.lm_head, QuantizedLinear)  # outside every block


def test_apply_quantization_layer_rules_compose_with_fallback():
    # first-match-wins across rules: a layer-specific rule plus a catch-all that
    # covers the remaining blocks and the head
    quantization = [
        _spec("int8", include=["*.mlp.*"], layer_idx=[1]),
        _spec("fp8", exclude=[]),
    ]
    model = _Stacked(n_layers=3)
    apply_quantization(model, _cfg(quantization, n_layers=3))
    for i, block in enumerate(model.blocks):
        attn_fwd = block.attn["q_proj"].quantization_config.dtype["weight"]["fwd"]
        assert attn_fwd == "fp8_e4m3"  # fallback covers attn everywhere
        mlp_fwd = block.mlp["down_proj"].quantization_config.dtype["weight"]["fwd"]
        assert mlp_fwd == ("int8" if i == 1 else "fp8_e4m3")
