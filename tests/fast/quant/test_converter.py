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
from src.quant.rotation import build_rotation_key
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


def _cfg(quantization):
    # QuantizationConfig resolves recipes by popping them out of the dict it is
    # handed, so a case table entry must not be the dict it consumes.
    quantization = copy.deepcopy(quantization)
    return TrainConfig(
        model=ModelConfig(
            d_model=32,
            n_layers=1,
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


def test_apply_quantization_owns_rotation_once_at_model_root():
    """Catch a rotation being registered per quantized module instead of once."""
    model = _Dense()
    config = _cfg(
        _spec(
            "int8",
            rotation={
                "rotation_cls": "hadamard",
                "rotation_kwargs": {"block_size": 16, "random_sign": True},
                "gemms": ["fwd"],
            },
        )
    )
    cfg = config.training.quantization[0]
    key = build_rotation_key(cfg.rotation, cfg.include, cfg.exclude)

    apply_quantization(model, config)
    model.to("meta")

    root_rotation = model.quant_rotations[key]
    assert root_rotation.sign_vector.device.type == "meta"
    assert model.attn["q_proj"].rotation is root_rotation
    assert model.mlp["down_proj"].rotation is root_rotation
    rotation_state = [k for k in model.state_dict() if "sign_vector" in k]
    assert rotation_state == [f"quant_rotations.{key}.sign_vector"]


def test_apply_quantization_gives_each_rule_its_own_rotation():
    """Catch one rule's rotation leaking into the modules claimed by another."""
    model = _Dense()
    config = _cfg(
        [
            _spec(
                "int8",
                include=["*attn*"],
                exclude=[],
                rotation={
                    "rotation_cls": "hadamard",
                    "rotation_kwargs": {"block_size": 16, "seed": 1},
                    "gemms": ["fwd"],
                },
            ),
            _spec(
                "int8",
                include=["*mlp*"],
                exclude=[],
                rotation={
                    "rotation_cls": "hadamard",
                    "rotation_kwargs": {"block_size": 32, "seed": 2},
                    "gemms": ["wgrad"],
                },
            ),
        ]
    )
    attn_cfg, mlp_cfg = config.training.quantization
    attn_key = build_rotation_key(attn_cfg.rotation, attn_cfg.include, attn_cfg.exclude)
    mlp_key = build_rotation_key(mlp_cfg.rotation, mlp_cfg.include, mlp_cfg.exclude)

    apply_quantization(model, config)

    assert attn_key != mlp_key
    assert sorted(model.quant_rotations) == sorted([attn_key, mlp_key])
    assert model.attn["q_proj"].rotation is model.quant_rotations[attn_key]
    assert model.mlp["down_proj"].rotation is model.quant_rotations[mlp_key]
    assert model.attn["q_proj"].rotation.block_size == 16
    assert model.mlp["down_proj"].rotation.block_size == 32
    assert sorted(k for k in model.state_dict() if "sign_vector" in k) == sorted(
        [
            f"quant_rotations.{attn_key}.sign_vector",
            f"quant_rotations.{mlp_key}.sign_vector",
        ]
    )


def test_apply_quantization_without_rotation_leaves_state_dict_clean():
    """A non-rotating run must not gain a rotation registry."""
    model = _Dense()

    apply_quantization(model, _cfg(_spec("int8")))

    assert not hasattr(model, "quant_rotations")
    assert not any("sign_vector" in k for k in model.state_dict())
