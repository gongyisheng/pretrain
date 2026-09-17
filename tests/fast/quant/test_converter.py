import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.layers.mlp import SparseMoEBlock
from src.model import build_model
from src.quant.convert import apply_quantization, enable_quantization
from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock
from src.quant.rotation import build_rotation_key
from src.utils.config import (
    ModelConfig,
    QuantizationConfig,
    TrainConfig,
    TrainingConfig,
)

from tests.fast.quant.helper import (
    FP8_E4M3_W8A8_E5M2_G8_DTYPES,
    FP8_E4M3_W8A8G8_DTYPES,
    INT8_W8A16_DTYPES,
    INT7_W8A16_DTYPES,
    INT6_W8A16_DTYPES,
    INT5_W8A16_DTYPES,
    INT4_W8A16_DTYPES,
)

INT_DTYPES = [
    INT8_W8A16_DTYPES,
    INT7_W8A16_DTYPES,
    INT6_W8A16_DTYPES,
    INT5_W8A16_DTYPES,
    INT4_W8A16_DTYPES,
]
ROWWISE_SCALE = {
    "weight": {"granularity": "rowwise", "block_shape": (1, 0)},
    "act": {"granularity": "rowwise", "block_shape": (1, 0)},
    "grad_out": {"granularity": "rowwise", "block_shape": (1, 0)},
    "scale_dtype": "fp32",
    "enable_global_scale": False,
}


def _spec(
    dtype=None,
    scale=None,
    include=None,
    exclude=None,
    rotation=None,
    enabled_after_steps=0,
):
    spec = {
        "enabled": True,
        "dtype": copy.deepcopy(
            FP8_E4M3_W8A8_E5M2_G8_DTYPES if dtype is None else dtype
        ),
    }
    if enabled_after_steps:
        spec["enabled_after_steps"] = enabled_after_steps
    if scale is not None:
        spec["scale"] = copy.deepcopy(scale)
    if include is not None:
        spec["include"] = include
    if exclude is not None:
        spec["exclude"] = exclude
    if rotation is not None:
        spec["rotation"] = rotation
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
    (
        _spec(FP8_E4M3_W8A8_E5M2_G8_DTYPES),
        ("attn.q_proj", "mlp.down_proj"),
        ("lm_head", "token_emb"),
    ),
    (
        _spec(
            FP8_E4M3_W8A8G8_DTYPES,
            scale={
                "weight": {"granularity": "blockwise", "block_shape": [1, 32]},
                "act": {"granularity": "blockwise", "block_shape": [1, 32]},
                "grad_out": {"granularity": "blockwise", "block_shape": [1, 32]},
                "scale_dtype": "fp8_e8m0",
                "enable_global_scale": False,
            },
        ),
        ("attn.q_proj", "mlp.down_proj"),
        ("lm_head", "token_emb"),
    ),
    # an include allowlist restricts the scope
    (
        _spec(include=["*attn*"], exclude=[]),
        ("attn.q_proj",),
        ("mlp.down_proj",),
    ),
    # multiple include patterns share the same quantization config
    (
        _spec(include=["*attn*", "*mlp*"], exclude=[]),
        ("attn.q_proj", "mlp.down_proj"),
        ("lm_head",),
    ),
    (
        _spec({"weight": "bf16", "act": "bf16", "grad_out": "bf16"}),
        (),
        ("attn.q_proj", "mlp.down_proj"),
    ),
    # the int8 series is weight-only but still swaps the module
    *[(_spec(dtype), ("attn.q_proj",), ("lm_head",)) for dtype in INT_DTYPES],
]


@pytest.mark.parametrize("quantization,swapped,untouched", APPLY_CASES)
def test_apply_quantization(quantization, swapped, untouched):
    model = _Dense()
    apply_quantization(model, _cfg(quantization))
    for fqn in swapped:
        assert isinstance(model.get_submodule(fqn), QuantizedLinear), fqn
    for fqn in untouched:
        assert not isinstance(model.get_submodule(fqn), QuantizedLinear), fqn


LAYER_CASES = [
    (None, [0, 1, 2]),
    ([], []),
    ([0], [0]),
    ([2, 0], [2, 0]),
    ([-1, -2], []),
    ([2, -1, 0, 2, -3, 0], [2, 0]),
    ([3, 20], []),
    ([3, 2, 0, 2, -1, 20], [2, 0]),
]
MLP_CLASSES = ["dense", "moe"]
COMPONENT_SCOPES = ["all", "attn", "mlp"]


@pytest.mark.parametrize("layer_idx,selected_layers", LAYER_CASES)
@pytest.mark.parametrize("mlp_cls", MLP_CLASSES)
@pytest.mark.parametrize("scope", COMPONENT_SCOPES)
def test_apply_quantization_layer_idx(layer_idx, selected_layers, mlp_cls, scope):
    mlp_kwargs = {"intermediate_size": 64}
    if mlp_cls == "moe":
        mlp_kwargs.update(
            n_routed_experts=4,
            n_routed_experts_per_token=2,
            n_shared_experts=1,
            aux_loss=True,
        )
    quantization = _spec(
        INT8_W8A16_DTYPES,
        include=[] if scope == "all" else [f"*{scope}*"],
        exclude=["k_proj", "*router.gate"],
        rotation={"rotation_cls": "hadamard"},
    )
    quantization["layer_idx"] = layer_idx
    config = TrainConfig(
        model=ModelConfig(
            d_model=32,
            n_layers=3,
            vocab_size=64,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 2}}],
            mlp=[{"mlp_cls": mlp_cls, "mlp_kwargs": mlp_kwargs}],
            tie_word_embeddings=False,
        ),
        training=TrainingConfig(quantization=quantization),
    )
    model = build_model(config)
    original_modules = dict(model.named_modules())
    original_parameters = dict(model.named_parameters())
    original_values = {
        name: parameter.detach().clone()
        for name, parameter in original_parameters.items()
    }

    assert apply_quantization(model, config) is model

    expected = set()
    for index in selected_layers:
        if scope in ("all", "attn"):
            expected.update(
                f"blocks.{index}.attn.{projection}"
                for projection in ("q_proj", "v_proj", "o_proj")
            )
        if scope in ("all", "mlp"):
            prefix = f"blocks.{index}.mlp"
            if mlp_cls == "moe":
                expected.add(prefix)
                prefix += ".shared_expert"
            expected.update(
                f"{prefix}.{projection}"
                for projection in ("gate_proj", "up_proj", "down_proj")
            )
    if layer_idx is None and scope == "all":
        expected.add("lm_head")
    quantized = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (QuantizedLinear, QuantizedSparseMoEBlock))
    }
    assert set(quantized) == expected
    for index in range(3):
        if index not in selected_layers:
            for name, module in original_modules.items():
                if name.startswith(f"blocks.{index}."):
                    assert model.get_submodule(name) is module
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original_values[name])
    if layer_idx is not None:
        assert model.lm_head is original_modules["lm_head"]
    if config.training.quantization.layer_idx == []:
        assert not hasattr(model, "quant_rotations")
    else:
        assert len(model.quant_rotations) == 1
        rotation = next(iter(model.quant_rotations.values()))
        assert all(module.rotation is rotation for module in quantized.values())
    assert all(not module.quantization_enabled for module in quantized.values())
    enable_quantization(model)
    assert all(module.quantization_enabled for module in quantized.values())


def test_apply_quantization_skips_a_tied_lm_head(capsys):
    # exclude nothing: the tie guard alone must keep lm_head unswapped, loudly
    model = _Dense(tie=True)
    apply_quantization(model, _cfg(_spec(exclude=[])))
    assert "tied" in capsys.readouterr().out
    assert not isinstance(model.lm_head, QuantizedLinear)
    assert isinstance(model.attn["q_proj"], QuantizedLinear)  # non-tied still swapped


MOE_CASES = [
    (_spec(scale=ROWWISE_SCALE), QuantizedSparseMoEBlock, False),
    (_spec(scale=ROWWISE_SCALE, exclude=["mlp"]), SparseMoEBlock, True),
]


@pytest.mark.parametrize("quantization,seam_owner,gate_swapped", MOE_CASES)
def test_apply_quantization_moe(quantization, seam_owner, gate_swapped):
    model = _MoE()
    apply_quantization(model, _cfg(quantization))
    assert isinstance(model.mlp.expert_gate, torch.nn.Parameter)
    assert isinstance(model.mlp.expert_up, torch.nn.Parameter)
    assert model.mlp.expert_mm.__func__ is seam_owner.expert_mm
    assert isinstance(model.mlp.router.gate, QuantizedLinear) is gate_swapped


def test_apply_quantization_attaches_no_metric_state():
    model = _Dense()
    apply_quantization(model, _cfg(_spec(FP8_E4M3_W8A8_E5M2_G8_DTYPES)))
    swapped = [m for m in model.modules() if isinstance(m, QuantizedLinear)]
    assert swapped and all(not hasattr(m, "quantization_probe") for m in swapped)


ENABLE_CASES = [
    ("dense", "attn.q_proj", "mlp.down_proj"),
    ("moe", "mlp", "mlp.router.gate"),
]


@pytest.mark.parametrize("model_kind,first_name,second_name", ENABLE_CASES)
def test_enable_quantization(model_kind, first_name, second_name):
    model = _Dense() if model_kind == "dense" else _MoE()
    config = _cfg(
        _spec(
            INT8_W8A16_DTYPES,
            include=[first_name, second_name],
            exclude=[],
            enabled_after_steps=2,
            rotation={
                "rotation_cls": "hadamard",
                "rotation_kwargs": {"block_size": 16, "seed": 1},
                "gemms": ["fwd"],
            },
        )
    )
    apply_quantization(model, config)
    first = model.get_submodule(first_name)
    second = model.get_submodule(second_name)
    assert not first.quantization_enabled
    assert not second.quantization_enabled
    parameters = tuple(model.parameters())
    rotation = first.rotation
    sign_vector = rotation.sign_vector.detach().clone()
    optimizer = torch.optim.AdamW(parameters)
    sum(parameter.square().sum() for parameter in parameters).backward()
    optimizer.step()
    parameter_values = {
        parameter: parameter.detach().clone() for parameter in parameters
    }
    state = {
        parameter: {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name, value in optimizer.state[parameter].items()
        }
        for parameter in parameters
    }

    enable_quantization(model)
    enable_quantization(model)
    assert first.quantization_enabled
    assert second.quantization_enabled
    assert first.rotation is rotation
    torch.testing.assert_close(rotation.sign_vector, sign_vector, rtol=0, atol=0)
    assert tuple(id(parameter) for parameter in model.parameters()) == tuple(
        id(parameter) for parameter in parameters
    )
    for parameter in parameters:
        torch.testing.assert_close(
            parameter, parameter_values[parameter], rtol=0, atol=0
        )
        for name, value in state[parameter].items():
            current = optimizer.state[parameter][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(current, value, rtol=0, atol=0)
            else:
                assert current == value


def test_apply_quantization_accepts_a_bare_config_namespace():
    # the converter only reads config.training.quantization, so a plain namespace of
    # an already-built config is enough — nothing else on TrainConfig is consulted
    model = nn.Sequential(nn.Linear(64, 64))
    rule = QuantizationConfig(
        enabled=True, dtype=dict(INT8_W8A16_DTYPES), include=["0"]
    )
    config = SimpleNamespace(training=SimpleNamespace(quantization=rule))
    apply_quantization(model, config)
    assert isinstance(model[0], QuantizedLinear)


def test_apply_quantization_owns_rotation_once_at_model_root():
    """Catch a rotation being registered per quantized module instead of once."""
    model = _Dense()
    config = _cfg(
        _spec(
            INT8_W8A16_DTYPES,
            rotation={
                "rotation_cls": "hadamard",
                "rotation_kwargs": {"block_size": 16, "random_sign": True},
                "gemms": ["fwd"],
            },
        )
    )
    cfg = config.training.quantization
    key = build_rotation_key(cfg.rotation, cfg.include, cfg.exclude)

    apply_quantization(model, config)
    model.to("meta")

    root_rotation = model.quant_rotations[key]
    assert root_rotation.sign_vector.device.type == "meta"
    assert model.attn["q_proj"].rotation is root_rotation
    assert model.mlp["down_proj"].rotation is root_rotation
    rotation_state = [k for k in model.state_dict() if "sign_vector" in k]
    assert rotation_state == [f"quant_rotations.{key}.sign_vector"]


def test_apply_quantization_without_rotation_leaves_state_dict_clean():
    """A non-rotating run must not gain a rotation registry."""
    model = _Dense()

    apply_quantization(model, _cfg(_spec(INT8_W8A16_DTYPES)))

    assert not hasattr(model, "quant_rotations")
    assert not any("sign_vector" in k for k in model.state_dict())
