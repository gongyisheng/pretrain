import copy
import glob
import json

import pytest
import torch
import yaml

from src.utils.config import (
    ModelConfig,
    TrainConfig,
    load_config,
    TrainingConfig,
    DataConfig,
    QuantizationConfig,
)


SCALE_TENSORS = ("weight", "act", "grad_out")
QUANTIZATION_ENABLED = [False, True]


def _scale_config(
    granularity="tensorwise",
    block_shape=None,
    scale_dtype=None,
    enable_global_scale=None,
):
    if block_shape is None and granularity != "blockwise":
        block_shape = (0, 0) if granularity == "tensorwise" else (1, 0)
    shapes = block_shape if isinstance(block_shape, dict) else {}
    scale = {
        tensor: {
            "granularity": granularity,
            "block_shape": shapes.get(tensor, block_shape),
        }
        for tensor in SCALE_TENSORS
    }
    if scale_dtype is not None:
        scale["scale_dtype"] = scale_dtype
    if enable_global_scale is not None:
        scale["enable_global_scale"] = enable_global_scale
    return scale


def _write_yaml(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


MINIMAL_CONFIG = {
    "max_seq_len": 128,
    "model": {
        "d_model": 64,
        "n_layers": 2,
        "vocab_size": 256,
        "attn": [
            {
                "attn_cls": "gqa",
                "attn_kwargs": {
                    "n_heads": 4,
                    "dropout": 0.0,
                    "attn_implementation": "flex_attention",
                },
            }
        ],
        "mlp": [
            {
                "mlp_cls": "dense",
                "mlp_kwargs": {
                    "activation_cls": "swiglu",
                    "intermediate_size": 0,
                },
            }
        ],
        "pos_emb_cls": "rope",
        "pos_emb_kwargs": {"rope_theta": 1e4},
    },
    "data": {
        "dataset": "test",
        "tokenizer_path": "tok",
        "data_dir": "data/",
        "val_split": 0.01,
        "num_workers": 0,
    },
    "training": {
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_steps": 10,
        "mixed_precision": "no",
        "grad_clip": 1.0,
        "checkpoint_dir": "ckpt/",
        "checkpoint_every": 5,
        "eval_every": 5,
        "eval_steps": 2,
    },
    "optimizer": {
        "optimizer_cls": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.1,
        "optimizer_kwargs": {"betas": [0.9, 0.95], "eps": 1e-8},
    },
    "scheduler": {"name": "cosine", "warmup_steps": 2, "min_lr": 1e-4},
    "logging": {"wandb_project": "test", "wandb_run_name": "test", "log_every": 1},
}


# ==================== ModelConfig defaults ====================


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.resolve_attn(0)[0] == "gqa"
    assert cfg.resolve_mlp(0)[0] == "dense"
    assert cfg.norm_cls == "rmsnorm"
    assert cfg.pos_emb_cls == "rope"
    assert cfg.residual_cls == "standard"
    # __post_init__ fills component defaults: attn_implementation for attn,
    # intermediate_size (4*d_model) and the activation pair for mlp.
    assert cfg.resolve_attn(0)[1] == {"attn_implementation": "flex_attention"}
    assert cfg.resolve_mlp(0)[1] == {
        "intermediate_size": 4 * 768,
        "activation_cls": "swiglu",
        "activation_kwargs": {},
    }
    assert cfg.norm_kwargs == {}
    assert cfg.pos_emb_kwargs == {}
    assert cfg.residual_kwargs == {}


# ==================== Loading from YAML ====================


def test_load_config(tmp_path):
    config = load_config(_write_yaml(tmp_path, MINIMAL_CONFIG))
    assert config.max_seq_len == 128
    assert config.model.n_layers == 2
    assert config.optimizer.lr == 1e-3
    assert config.model.resolve_attn(0)[1]["n_heads"] == 4
    assert config.model.resolve_mlp(0)[1]["activation_cls"] == "swiglu"
    assert config.model.pos_emb_kwargs["rope_theta"] == 1e4

    exported = config.to_dict()
    assert exported["max_seq_len"] == 128
    assert exported["model"]["attn"][0]["attn_cls"] == "gqa"
    assert exported["model"]["attn"][0]["attn_kwargs"]["n_heads"] == 4
    assert load_config(_write_yaml(tmp_path, exported)).to_dict() == exported


SCALE_DTYPE_CASES = [
    (_scale_config("rowwise", scale_dtype="fp32"), "fp32"),
    (_scale_config("blockwise", [1, 32], scale_dtype="fp8_e8m0"), "fp8_e8m0"),
]


@pytest.mark.parametrize("enabled", QUANTIZATION_ENABLED)
@pytest.mark.parametrize("case", SCALE_DTYPE_CASES)
def test_train_config_to_dict_scale_dtype(tmp_path, enabled, case):
    scale, scale_dtype = case
    config = TrainConfig(
        training=TrainingConfig(mixed_precision="no"),
        quantization=QuantizationConfig(
            enabled=enabled,
            dtype={"weight": "fp8_e4m3"},
            scale=copy.deepcopy(scale) if enabled else {"scale_dtype": scale_dtype},
        ),
    )
    exported = config.to_dict()
    assert config.quantization.scale["scale_dtype"] == scale_dtype
    assert exported["quantization"]["scale"]["scale_dtype"] == scale_dtype
    assert (
        json.loads(json.dumps(exported))["quantization"]["scale"]["scale_dtype"]
        == scale_dtype
    )
    restored = load_config(_write_yaml(tmp_path, exported))
    assert restored.quantization.scale["scale_dtype"] == scale_dtype


BLOCK_SHAPE_TENSORS = ["weight", "act", "grad_out"]
BLOCK_SHAPE_OVERRIDES = [(1, 16), (16, 16), (1, 32), (1, 64), (1, 128)]
OFF_BLOCKWISE_SHAPES = [
    {
        "weight": (1, 128),
        "act": (1, 128),
        "grad_out": (1, 128),
    },
    (1, 16),
    [1, 16],
    None,
]


@pytest.mark.parametrize("tensor", BLOCK_SHAPE_TENSORS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPE_OVERRIDES)
def test_config_to_dict_roundtrips_per_tensor_block_shape(
    tmp_path, tensor, block_shape
):
    expected_shapes = {"weight": (16, 16), "act": (1, 16), "grad_out": (1, 16)}
    expected_shapes[tensor] = block_shape
    config = TrainConfig(
        training=TrainingConfig(mixed_precision="no"),
        quantization=QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp4_e2m1", "act": "fp4_e2m1", "grad_out": "fp4_e2m1"},
            scale=_scale_config("blockwise", expected_shapes),
        ),
    )
    restored = load_config(_write_yaml(tmp_path, config.to_dict()))
    assert {
        tensor: restored.quantization.scale[tensor]["block_shape"]
        for tensor in SCALE_TENSORS
    } == expected_shapes


NONBLOCK_SCALE_CASES = [("tensorwise", (0, 0)), ("rowwise", (1, 0))]


@pytest.mark.parametrize("case", NONBLOCK_SCALE_CASES)
def test_config_to_dict_roundtrips_nonblock_scale_shape(case):
    granularity, expected_shape = case
    config = TrainConfig(
        training=TrainingConfig(mixed_precision="no"),
        quantization=QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
            scale={
                tensor: {"granularity": granularity, "block_shape": None}
                for tensor in SCALE_TENSORS
            },
        ),
    )
    exported = config.to_dict()
    exported_scale = exported["quantization"]["scale"]
    assert {
        tensor: exported_scale[tensor]["block_shape"] for tensor in SCALE_TENSORS
    } == {tensor: expected_shape for tensor in SCALE_TENSORS}

    for serialized, deserialize in (
        (json.dumps(exported), json.loads),
        (yaml.safe_dump(exported), yaml.safe_load),
    ):
        decoded = deserialize(serialized)
        assert {
            tensor: decoded["quantization"]["scale"][tensor]["block_shape"]
            for tensor in SCALE_TENSORS
        } == {tensor: list(expected_shape) for tensor in SCALE_TENSORS}
        restored = TrainConfig(
            training=TrainingConfig(**decoded["training"]),
            quantization=QuantizationConfig(**decoded["quantization"]),
        )
        assert {
            tensor: restored.quantization.scale[tensor]["block_shape"]
            for tensor in SCALE_TENSORS
        } == {tensor: expected_shape for tensor in SCALE_TENSORS}


def test_quantization_config_rejects_runtime_scale_dtype():
    with pytest.raises(ValueError):
        QuantizationConfig(enabled=True, scale={"scale_dtype": torch.float32})


def test_config_to_dict_rotation_roundtrip(tmp_path):
    config = TrainConfig(
        training=TrainingConfig(mixed_precision="no"),
        quantization=QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
            rotation={
                "rotation_cls": "hadamard",
                "rotation_kwargs": {
                    "block_size": 4,
                    "random_sign": True,
                    "sign_vector": [1.0, -1.0, -1.0, 1.0],
                },
                "rotation_axes": {
                    "weight": {"fwd": [-1], "dgrad": [-1]},
                    "act": {"fwd": [-2], "wgrad": [-2]},
                },
            },
        ),
    )
    original_rule = config.quantization

    exported = config.to_dict()
    exported_rotation = exported["quantization"]["rotation"]
    assert set(exported_rotation) == {
        "rotation_cls",
        "rotation_kwargs",
        "rotation_axes",
    }
    assert exported_rotation["rotation_kwargs"]["sign_vector"] == [
        1.0,
        -1.0,
        -1.0,
        1.0,
    ]
    json.loads(json.dumps(exported))
    restored = load_config(_write_yaml(tmp_path, exported))
    restored_rule = restored.quantization
    assert restored_rule.rotation == original_rule.rotation


# ==================== CLI overrides ====================


def test_load_config_optimizer_defaults(tmp_path):
    raw = copy.deepcopy(MINIMAL_CONFIG)
    del raw["optimizer"]
    config = load_config(_write_yaml(tmp_path, raw)).optimizer
    assert (config.optimizer_cls, config.lr) == ("adamw", 5e-4)
    assert config.optimizer_kwargs == {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "fused": True,
    }


@pytest.mark.parametrize("muon_cls", ["muon", "muonc"])
@pytest.mark.parametrize("adam_cls", ["adamw", "adamc"])
def test_load_config_optimizer_nested_kwargs(tmp_path, muon_cls, adam_cls):
    raw = copy.deepcopy(MINIMAL_CONFIG)
    raw["optimizer"] = {
        "optimizer_cls": "muonadam",
        "lr": 1e-3,
        "weight_decay": 0.1,
        "optimizer_kwargs": {
            "muon_cls": muon_cls,
            "muon_kwargs": {"eps": "1e-6"},
            "adam_cls": adam_cls,
            "adam_kwargs": {"eps": "1e-7"},
        },
    }
    config = load_config(_write_yaml(tmp_path, raw)).optimizer
    assert config.optimizer_kwargs["muon_cls"] == muon_cls
    assert config.optimizer_kwargs["muon_kwargs"]["eps"] == 1e-6
    assert config.optimizer_kwargs["adam_cls"] == adam_cls
    assert config.optimizer_kwargs["adam_kwargs"]["eps"] == 1e-7


def test_load_config_overrides(tmp_path):
    config_data = copy.deepcopy(MINIMAL_CONFIG)
    config_data["quantization"] = {
        "enabled": True,
        "dtype": {"weight": "fp8_e4m3"},
    }
    config = load_config(
        _write_yaml(tmp_path, config_data),
        overrides=[
            "optimizer.lr=3e-4",
            "training.batch_size=8",
            "quantization.enabled_after_steps=2",
            "quantization.enabled_before_steps=3",
            "optimizer.lr_mult.token_emb=3e-4",
            "seed=23",
            "model.pos_emb_kwargs.rope_theta=5000",
        ],
    )
    assert config.optimizer.lr == 3e-4
    assert config.training.batch_size == 8
    assert config.quantization.enabled_after_steps == 2
    assert config.quantization.enabled_before_steps == 3
    assert isinstance(config.quantization.enabled_before_steps, int)
    assert config.optimizer.lr_mult["token_emb"] == 3e-4
    assert config.seed == 23
    assert config.model.pos_emb_kwargs["rope_theta"] == 5000

    config_data["quantization"]["enabled_before_steps"] = 3
    config = load_config(
        _write_yaml(tmp_path, config_data),
        overrides=["quantization.enabled_before_steps=null"],
    )
    assert config.quantization.enabled_before_steps is None


# ==================== Nested coercion ====================


def test_load_config_coerces_nested_kwargs(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  pos_emb_kwargs:\n    rope_theta: 1e4\n")
    cfg = load_config(str(p))
    assert cfg.model.pos_emb_kwargs["rope_theta"] == 10000.0


# ==================== TrainingConfig / DataConfig ====================


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.intra_doc_masking is True
    assert cfg.eval_train is False
    assert cfg.device == "auto"


def test_data_config_packing_default():
    cfg = DataConfig()
    assert cfg.packing is True


def test_intra_doc_masking_yaml_override(tmp_path):
    yaml_content = """
max_seq_len: 128
training:
  intra_doc_masking: false
data:
  packing: false
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.training.intra_doc_masking is False
    assert cfg.data.packing is False


def test_unknown_yaml_fields_ignored(tmp_path):
    yaml_content = """
max_seq_len: 128
data:
  eot_token_id: 0
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.max_seq_len == 128


# ==================== TokenizerTrainingConfig ====================


def test_tokenizer_training_defaults():
    from src.utils.config import TokenizerTrainingConfig

    tc = TokenizerTrainingConfig()
    assert tc.method == "bpe"
    # eval_num_docs default is filled by __post_init__
    assert tc.method_kwargs == {"eval_num_docs": 1000}
    assert tc.num_samples == 1_000_000
    assert tc.checkpoint_every == 5000
    assert tc.eval_every == 5000


def test_loads_superbpe_tokenizer_training_yaml(tmp_path):
    yaml_content = """
model:
  vocab_size: 200000
data:
  dataset: openwebtext
  tokenizer_path: tokenizers/superbpe_200k_t80k
tokenizer_training:
  method: superbpe
  method_kwargs:
    transition_size: 80000
    max_superword_words: 4
  checkpoint_dir: tokenizers/superbpe_200k_t80k
  checkpoint_every: 5000
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.data.tokenizer_path == "tokenizers/superbpe_200k_t80k"
    assert cfg.tokenizer_training.method == "superbpe"
    assert cfg.tokenizer_training.method_kwargs == {
        "transition_size": 80000,
        "max_superword_words": 4,
        "eval_num_docs": 1000,  # filled by __post_init__
    }
    assert cfg.tokenizer_training.checkpoint_dir == "tokenizers/superbpe_200k_t80k"


# ==================== task and eval_train fields ====================


TASK_CASES = [({}, "pretrain"), ({"task": "sft"}, "sft")]


@pytest.mark.parametrize("case", TASK_CASES)
def test_train_config_task(tmp_path, case):
    kwargs, expected = case
    assert TrainConfig(**kwargs).task == expected
    assert load_config(_write_yaml(tmp_path, kwargs)).task == expected


# ==================== attn ====================


ATTN_CLASSES = ["gqa", "mha", "mla"]


@pytest.mark.parametrize("attn_cls", ATTN_CLASSES)
def test_model_config_attn(attn_cls):
    config = ModelConfig(attn=[{"attn_cls": attn_cls, "attn_kwargs": {"n_heads": 8}}])
    resolved_cls, kwargs = config.resolve_attn(0)
    assert resolved_cls == attn_cls
    assert kwargs["n_heads"] == 8
    assert kwargs["attn_sink"] is False


@pytest.mark.parametrize("attn_cls", ATTN_CLASSES)
def test_model_config_attn_sink(tmp_path, attn_cls):
    cfg = ModelConfig(
        d_model=64,
        attn=[
            {
                "attn_cls": attn_cls,
                "attn_kwargs": {"n_heads": 4, "attn_sink": True},
            }
        ],
    )
    assert cfg.resolve_attn(0)[1]["attn_sink"] is True

    config_data = copy.deepcopy(MINIMAL_CONFIG)
    config_data["model"]["attn"][0]["attn_cls"] = attn_cls
    config_data["model"]["attn"][0]["attn_kwargs"]["attn_sink"] = True
    config = load_config(_write_yaml(tmp_path, config_data))
    exported = config.to_dict()
    assert exported["model"]["attn"][0]["attn_kwargs"]["attn_sink"] is True
    assert load_config(_write_yaml(tmp_path, exported)).to_dict() == exported


@pytest.mark.parametrize("attn_cls", ATTN_CLASSES)
def test_model_config_attn_sink_raise_error(tmp_path, attn_cls):
    kwargs = {"attn_implementation": "sdpa", "attn_sink": True, "n_heads": 4}
    attn = [{"attn_cls": attn_cls, "attn_kwargs": kwargs}]
    with pytest.raises(ValueError):
        ModelConfig(attn=attn)
    with pytest.raises(ValueError):
        load_config(_write_yaml(tmp_path, {"model": {"attn": attn}}))


def test_attn_kwargs_round_trip_from_yaml(tmp_path):
    yaml_content = """
model:
  arch: qwen3
  d_model: 64
  attn:
    - attn_cls: mla
      attn_kwargs:
        n_heads: 8
        kv_lora_rank: 32
        qk_rope_head_dim: 16
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.model.resolve_attn(0)[0] == "mla"
    assert cfg.model.resolve_attn(0)[1]["kv_lora_rank"] == 32
    assert cfg.model.resolve_attn(0)[1]["qk_rope_head_dim"] == 16
    assert cfg.model.resolve_attn(0)[1]["n_heads"] == 8


# ==================== per-layer attn schema + resolver ====================


def test_attn_single_item_covers_all_layers():
    cfg = ModelConfig(
        d_model=64,
        n_layers=4,
        attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}}],
    )
    assert [cfg.resolve_attn(i)[0] for i in range(cfg.n_layers)] == ["gqa"] * 4
    assert cfg.resolve_attn(0)[1]["n_kv_heads"] == 4  # per-item defaulting ran
    assert cfg.resolve_attn(0)[1]["attn_implementation"] == "flex_attention"


def test_attn_mixed_per_layer_complement():
    cfg = ModelConfig(
        d_model=64,
        n_layers=4,
        attn=[
            {
                "attn_cls": "mha",
                "attn_kwargs": {"n_heads": 4, "attn_sink": True},
                "layer_idx": [0],
            },
            {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4, "n_kv_heads": 2}},
        ],
    )
    assert [cfg.resolve_attn(i)[0] for i in range(cfg.n_layers)] == [
        "mha",
        "gqa",
        "gqa",
        "gqa",
    ]
    assert cfg.attn[1]["layer_idx"] == [1, 2, 3]
    assert [cfg.resolve_attn(layer)[1]["attn_sink"] for layer in range(4)] == [
        True,
        False,
        False,
        False,
    ]


def test_attn_conflict_raises():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=4,
            attn=[
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}, "layer_idx": [0]},
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}, "layer_idx": [0, 1]},
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}},
            ],
        )


def test_attn_gap_raises():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=4,
            attn=[
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}, "layer_idx": [0]},
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}, "layer_idx": [1]},
            ],
        )


def test_attn_two_bare_items_raise():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=4,
            attn=[
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}},
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}},
            ],
        )


def test_attn_dup_index_within_item_raises():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=4,
            attn=[
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}, "layer_idx": [0, 0]},
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}},
            ],
        )


def test_model_config_mla_defaults():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        attn=[{"attn_cls": "mla", "attn_kwargs": {"n_heads": 4}}],
    )
    kw = cfg.resolve_attn(0)[1]
    assert kw["qk_nope_head_dim"] == 16 and kw["qk_rope_head_dim"] == 8
    assert (
        kw["v_head_dim"] == 16 and kw["kv_lora_rank"] == 64 and kw["q_lora_rank"] == 0
    )


def test_attn_implementation_must_be_shared_across_layers():
    # The trainer builds one attention mask shared across layers, so layers
    # cannot disagree on attn_implementation.
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=2,
            attn=[
                {
                    "attn_cls": "gqa",
                    "attn_kwargs": {"n_heads": 4, "attn_implementation": "sdpa"},
                    "layer_idx": [0],
                },
                {
                    "attn_cls": "gqa",
                    "attn_kwargs": {
                        "n_heads": 4,
                        "attn_implementation": "flex_attention",
                    },
                },
            ],
        )


def test_attn_implementation_shared_when_uniform():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        attn=[
            {
                "attn_cls": "gqa",
                "attn_kwargs": {"n_heads": 4, "attn_implementation": "sdpa"},
            }
        ],
    )
    assert cfg.attn_implementation == "sdpa"


# ==================== component defaults + validation (moved into config) ====================


@pytest.mark.parametrize("mlp_cls", ["dense", "moe"])
def test_modelconfig_resolves_intermediate_size(mlp_cls):
    extra = (
        {"n_routed_experts": 4, "aux_loss": True, "aux_loss_coef": 1e-3}
        if mlp_cls == "moe"
        else {}
    )
    cfg = ModelConfig(
        d_model=128, mlp=[{"mlp_cls": mlp_cls, "mlp_kwargs": dict(extra)}]
    )
    assert cfg.resolve_mlp(0)[1]["intermediate_size"] == 4 * 128
    # explicit value preserved
    cfg2 = ModelConfig(
        d_model=128,
        mlp=[
            {
                "mlp_cls": mlp_cls,
                "mlp_kwargs": {"intermediate_size": 256, **extra},
            }
        ],
    )
    assert cfg2.resolve_mlp(0)[1]["intermediate_size"] == 256


def test_model_config_moe_defaults():
    # aux_loss on: expert_bias defaults off, aux_loss_coef defaulted
    cfg = ModelConfig(
        d_model=64,
        mlp=[
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {"n_routed_experts": 4, "aux_loss": True},
            }
        ],
    )
    assert cfg.resolve_mlp(0)[1]["expert_bias"] is False
    assert cfg.resolve_mlp(0)[1]["aux_loss"] is True
    assert cfg.resolve_mlp(0)[1]["aux_loss_coef"] == 0.001
    assert cfg.resolve_mlp(0)[1]["router_score_fn"] == "sigmoid"
    assert cfg.resolve_mlp(0)[1]["latent_moe"] is False
    assert "latent_dim" not in cfg.resolve_mlp(0)[1]
    # expert_bias on: aux_loss stays off, bias update rate defaulted
    cfg2 = ModelConfig(
        d_model=64,
        mlp=[
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {"n_routed_experts": 4, "expert_bias": True},
            }
        ],
    )
    assert cfg2.resolve_mlp(0)[1]["aux_loss"] is False
    assert cfg2.resolve_mlp(0)[1]["expert_bias_update_rate"] == 0.001
    assert "aux_loss_coef" not in cfg2.resolve_mlp(0)[1]


def test_modelconfig_moe_aux_loss_and_expert_bias_mutually_exclusive():
    # both on
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            mlp=[
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "n_routed_experts": 4,
                        "aux_loss": True,
                        "expert_bias": True,
                    },
                }
            ],
        )
    # both off (defaults) — must opt into exactly one
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            mlp=[{"mlp_cls": "moe", "mlp_kwargs": {"n_routed_experts": 4}}],
        )


def test_modelconfig_moe_router_score_fn_softmax_kept():
    cfg = ModelConfig(
        d_model=64,
        mlp=[
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {
                    "n_routed_experts": 4,
                    "aux_loss": True,
                    "router_score_fn": "softmax",
                },
            }
        ],
    )
    assert cfg.resolve_mlp(0)[1]["router_score_fn"] == "softmax"


def test_modelconfig_moe_unknown_router_score_fn_raises():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            mlp=[
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "n_routed_experts": 4,
                        "aux_loss": True,
                        "router_score_fn": "argmax",
                    },
                }
            ],
        )


ACT_KWARGS_ERRORS = [
    {
        "activation_cls": "swiglu",
        "activation_kwargs": {"act_limit": {"up": {"min": 7, "max": 1}}},
    },
    {
        "activation_cls": "swiglu",
        "activation_kwargs": {"act_limit": {"up": {"min": 0, "max": -1}}},
    },
    {
        "activation_cls": "silu",
        "activation_kwargs": {"act_limit": {"up": {"min": 7, "max": 1}}},
    },
]

ACT_KWARGS_CASES = [
    {"activation_cls": "swiglu", "activation_kwargs": {"act_limit": {"max": 7}}},
    {
        "activation_cls": "silu",
        "activation_kwargs": {"act_limit": {"gate": {"max": 7}}},
    },
    {"activation_cls": "swiglu", "activation_kwargs": {"act_limit": {"gate": 7}}},
    {"activation_cls": "swiglu", "activation_kwargs": {"act_limit": {"gate": {}}}},
    {
        "activation_cls": "swiglu",
        "activation_kwargs": {"act_limit": {"gate": {"lo": 1}}},
    },
    {
        "activation_cls": "swiglu",
        "activation_kwargs": {"act_limit": {"up": {"min": "x"}}},
    },
    {
        "activation_cls": "swiglu",
        "activation_kwargs": {"act_limit": {"gate": {"max": 7}}},
    },
    {
        "activation_cls": "swiglu",
        "activation_kwargs": {
            "act_limit": {"gate": {"max": 7}, "up": {"min": -7, "max": 7}}
        },
    },
    {
        "activation_cls": "swiglu",
        "activation_kwargs": {"act_limit": {"up": {"min": -7}}},
    },
    {"activation_cls": "silu", "activation_kwargs": {"act_limit": {"up": {"max": 7}}}},
    {
        "activation_cls": "silu",
        "activation_kwargs": {"act_limit": {"up": {"min": -7, "max": 7}}},
    },
    {"activation_cls": "swiglu", "activation_kwargs": {"alpha": 1.702}},
    {"activation_cls": "swiglu", "activation_kwargs": {"up_shift": 1.0}},
    {
        "activation_cls": "swiglu",
        "activation_kwargs": {
            "alpha": 1.702,
            "up_shift": 1.0,
            "act_limit": {"gate": {"max": 7.0}, "up": {"min": -7.0, "max": 7.0}},
        },
    },
]
ACTIVATION_CLASSES = ["swiglu", "silu", "bilinear", "powlu"]
MLP_CLASSES = ["dense", "moe"]


def test_modelconfig_activation_cls_raise_error():
    with pytest.raises(ValueError):
        ModelConfig(
            mlp=[{"mlp_cls": "dense", "mlp_kwargs": {"activation_cls": "mish"}}]
        )


@pytest.mark.parametrize("activation_cls", ACTIVATION_CLASSES)
def test_modelconfig_activation_cls(activation_cls):
    cfg = ModelConfig(
        d_model=64,
        mlp=[{"mlp_cls": "dense", "mlp_kwargs": {"activation_cls": activation_cls}}],
    )
    assert cfg.resolve_mlp(0)[1]["activation_cls"] == activation_cls


@pytest.mark.parametrize("mlp_cls", MLP_CLASSES)
@pytest.mark.parametrize("kwargs", ACT_KWARGS_ERRORS)
def test_modelconfig_activation_kwargs_raise_error(mlp_cls, kwargs):
    kwargs = copy.deepcopy(kwargs)
    if mlp_cls == "moe":
        kwargs |= {"n_routed_experts": 4, "aux_loss": True}
    with pytest.raises(ValueError):
        ModelConfig(d_model=64, mlp=[{"mlp_cls": mlp_cls, "mlp_kwargs": kwargs}])


@pytest.mark.parametrize("act_kwargs", [7.0, "off", None])
def test_modelconfig_activation_kwargs_non_mapping_dropped(act_kwargs):
    """A non-dict activation_kwargs is discarded, leaving the block's own default."""
    cfg = ModelConfig(
        d_model=64,
        mlp=[{"mlp_cls": "dense", "mlp_kwargs": {"activation_kwargs": act_kwargs}}],
    )
    assert "activation_kwargs" not in cfg.resolve_mlp(0)[1]


@pytest.mark.parametrize("act_limit", [7.0, "off"])
def test_modelconfig_activation_kwargs_act_limit_non_dict_dropped(act_limit):
    """A limit that is not a dict is discarded, leaving the activation unclamped."""
    cfg = ModelConfig(
        d_model=64,
        mlp=[
            {
                "mlp_cls": "dense",
                "mlp_kwargs": {"activation_kwargs": {"act_limit": act_limit}},
            }
        ],
    )
    assert cfg.resolve_mlp(0)[1]["activation_kwargs"] == {}


@pytest.mark.parametrize("kwargs", ACT_KWARGS_CASES)
def test_modelconfig_activation_kwargs(kwargs):
    cfg = ModelConfig(
        d_model=64,
        mlp=[{"mlp_cls": "dense", "mlp_kwargs": copy.deepcopy(kwargs)}],
    )
    assert cfg.resolve_mlp(0)[1]["activation_kwargs"] == kwargs["activation_kwargs"]


def test_modelconfig_validates_attn_dims():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=100, attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 3}}]
        )
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            attn=[
                {
                    "attn_cls": "gqa",
                    "attn_kwargs": {"n_heads": 4, "n_kv_heads": 3},
                }
            ],
        )


# ==================== string-field validation ====================


def test_scheduler_unknown_name_raises():
    from src.utils.config import SchedulerConfig

    with pytest.raises(ValueError):
        SchedulerConfig(name="step")
    SchedulerConfig(name="cosine")
    SchedulerConfig(name="constant")


OPTIMIZER_DEFAULT_CASES = [
    ("adamw", {"betas": (0.9, 0.95), "eps": 1e-8, "fused": True}),
    ("adamc", {"betas": (0.9, 0.95), "eps": 1e-8, "fused": True}),
    ("lion", {"betas": (0.9, 0.99), "foreach": True}),
    (
        "muonadam",
        {
            "muon_cls": "muon",
            "muon_kwargs": {
                "momentum": 0.95,
                "nesterov": True,
                "adjust_lr_fn": "match_rms_adamw",
                "eps": 1e-8,
            },
            "adam_cls": "adamw",
            "adam_kwargs": {
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "fused": True,
            },
        },
    ),
]


@pytest.mark.parametrize("case", OPTIMIZER_DEFAULT_CASES)
def test_optimizer_config_defaults(case):
    from src.utils.config import OptimizerConfig

    optimizer_cls, expected = case
    # Unset kwargs get the pretraining-tuned defaults for the selected optimizer.
    assert OptimizerConfig(optimizer_cls, lr=1e-3).optimizer_kwargs == expected
    if optimizer_cls == "muonadam":
        return
    # An explicit value wins; the remaining keys are still filled in.
    for key in expected:
        cfg = OptimizerConfig(optimizer_cls, lr=1e-3, optimizer_kwargs={key: "set"})
        assert cfg.optimizer_kwargs == {**expected, key: "set"}


@pytest.mark.parametrize("muon_cls", ["muon", "muonc"])
@pytest.mark.parametrize("adam_cls", ["adamw", "adamc"])
def test_optimizer_config_muon_nested_kwargs(muon_cls, adam_cls):
    from src.utils.config import OptimizerConfig

    cfg = OptimizerConfig(
        "muonadam",
        lr=1e-3,
        optimizer_kwargs={
            "muon_cls": muon_cls,
            "muon_kwargs": {"momentum": 0.8, "ns_steps": 3},
            "adam_cls": adam_cls,
            "adam_kwargs": {"betas": (0.8, 0.9), "fused": False},
        },
    )

    assert cfg.optimizer_kwargs == {
        "muon_cls": muon_cls,
        "muon_kwargs": {
            "momentum": 0.8,
            "ns_steps": 3,
            "nesterov": True,
            "adjust_lr_fn": "match_rms_adamw",
            "eps": 1e-8,
        },
        "adam_cls": adam_cls,
        "adam_kwargs": {"betas": (0.8, 0.9), "eps": 1e-8, "fused": False},
    }


INVALID_MUON_KWARGS_CASES = [
    (
        "muon_kwargs",
        None,
        {
            "momentum": 0.95,
            "nesterov": True,
            "adjust_lr_fn": "match_rms_adamw",
            "eps": 1e-8,
        },
    ),
    ("adam_kwargs", "invalid", {"betas": (0.9, 0.95), "eps": 1e-8, "fused": True}),
]


@pytest.mark.parametrize("case", INVALID_MUON_KWARGS_CASES)
def test_optimizer_config_muon_nested_kwargs_invalid(case):
    from src.utils.config import OptimizerConfig

    key, value, expected = case
    cfg = OptimizerConfig("muonadam", lr=1e-3, optimizer_kwargs={key: value})

    assert cfg.optimizer_kwargs[key] == expected


def test_optimizer_config_raise_error():
    from src.utils.config import OptimizerConfig

    with pytest.raises(ValueError):
        OptimizerConfig("sgd", lr=1e-3)
    with pytest.raises(ValueError):
        OptimizerConfig("muonadam", lr=1e-3, optimizer_kwargs={"muon_cls": "adamc"})
    with pytest.raises(ValueError):
        OptimizerConfig("muonadam", lr=1e-3, optimizer_kwargs={"adam_cls": "lion"})


TRAINING_CONFIG_ERRORS = [
    {"mixed_precision": "fp8"},
    {"loss_fn": "huber"},
    {"device": "tpu"},
]


@pytest.mark.parametrize("kwargs", TRAINING_CONFIG_ERRORS)
def test_training_config_raise_error(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_training_config_device():
    assert TrainingConfig(device="cpu").device == "cpu"


# ==================== dropless MoE + precision guard ====================


def _moe_train_config(mixed_precision):
    """Build a minimal TrainConfig with dropless MoE."""
    mlp_kwargs = {
        "n_routed_experts": 4,
        "n_routed_experts_per_token": 2,
        "intermediate_size": 64,
        "aux_loss": True,
        "aux_loss_coef": 1e-3,
    }
    return TrainConfig(
        model=ModelConfig(
            d_model=64, mlp=[{"mlp_cls": "moe", "mlp_kwargs": mlp_kwargs}]
        ),
        training=TrainingConfig(mixed_precision=mixed_precision),
    )


def test_dropless_moe_without_mixed_precision_raises():
    with pytest.raises(ValueError):
        _moe_train_config("no")


@pytest.mark.parametrize("mixed_precision", ["bf16", "fp16"])
def test_dropless_moe_reduced_precision_ok(mixed_precision):
    cfg = _moe_train_config(mixed_precision)
    assert cfg.training.mixed_precision == mixed_precision


# ==================== Quantization config ====================


DISABLED_QUANT_ROTATIONS = [
    {},
    {"rotation_cls": "unknown", "rotation_kwargs": 4, "rotation_axes": [-3]},
]


@pytest.mark.parametrize("rotation", DISABLED_QUANT_ROTATIONS)
def test_quant_defaults_empty(rotation):
    expected_rotation = {
        "rotation_cls": None,
        "rotation_kwargs": {},
        "rotation_axes": {},
    }
    expected_rotation.update(rotation)
    q = QuantizationConfig(rotation=rotation)
    assert q.enabled is False
    assert q.rounding == {}
    assert q.exclude == ["lm_head", "*mlp.router.gate"]
    # disabled rule is inert: no dtype/scale defaults applied
    assert q.dtype == {} and q.scale == {}
    assert q.rotation == expected_rotation


ROUNDING_CASES = [
    ({}, {"weight": "RNE", "act": "RNE", "grad_out": "RNE"}),
    ({"grad_out": "SR"}, {"weight": "RNE", "act": "RNE", "grad_out": "SR"}),
    ({"act": "SR"}, {"weight": "RNE", "act": "SR", "grad_out": "RNE"}),
]


@pytest.mark.parametrize("case", ROUNDING_CASES)
def test_train_config_quantization_rounding(case):
    rounding, expected = case
    config = TrainConfig(
        training=TrainingConfig(mixed_precision="bf16"),
        quantization=QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            rounding=copy.deepcopy(rounding),
        ),
    )
    assert config.quantization.rounding == expected
    assert config.quantization.dtype["act"] == {"fwd": "bf16", "wgrad": "bf16"}
    assert config.to_dict()["quantization"]["rounding"] == expected


ROUNDING_ERRORS = [{"grad_input": "SR"}, {"act": "stochastic"}]


@pytest.mark.parametrize("rounding", ROUNDING_ERRORS)
def test_quantization_config_rounding_raise_error(rounding):
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True, dtype={"weight": "fp8_e4m3"}, rounding=rounding
        )


def test_quant_tensor_defaults_follow_mixed_precision():
    r = TrainConfig(
        training=TrainingConfig(mixed_precision="bf16"),
        quantization=QuantizationConfig(enabled=True, dtype={"weight": "bf16"}),
    ).quantization
    # every slot defaults to the compute dtype
    assert r.dtype == {
        "weight": {"fwd": "bf16", "dgrad": "bf16"},
        "act": {"fwd": "bf16", "wgrad": "bf16"},
        "grad_out": {"dgrad": "bf16", "wgrad": "bf16"},
    }
    r32 = TrainConfig(
        training=TrainingConfig(mixed_precision="no"),
        quantization=QuantizationConfig(enabled=True, dtype={"weight": "fp32"}),
    ).quantization
    assert r32.dtype["weight"]["fwd"] == "fp32"


def test_quant_dtype_scalar_applies_to_every_consuming_gemm():
    r = TrainConfig(
        training=TrainingConfig(mixed_precision="bf16"),
        quantization=QuantizationConfig(enabled=True, dtype={"weight": "fp8_e4m3"}),
    ).quantization
    assert r.dtype == {
        "weight": {"fwd": "fp8_e4m3", "dgrad": "fp8_e4m3"},
        "act": {"fwd": "bf16", "wgrad": "bf16"},
        "grad_out": {"dgrad": "bf16", "wgrad": "bf16"},
    }


def test_quant_dtype_scopes_a_tensor_per_gemm():
    # the w16a16dx8 cell: grad_out quantized in dgrad only
    q = QuantizationConfig(
        enabled=True,
        dtype={"grad_out": {"dgrad": "fp8_e4m3", "wgrad": "bf16"}},
    )
    assert q.dtype["grad_out"] == {"dgrad": "fp8_e4m3", "wgrad": "bf16"}


def test_quant_scoped_dtype_leaves_unset_gemm_to_mixed_precision():
    r = TrainConfig(
        training=TrainingConfig(mixed_precision="bf16"),
        quantization=QuantizationConfig(
            enabled=True, dtype={"grad_out": {"dgrad": "int8"}}
        ),
    ).quantization
    assert r.dtype["grad_out"] == {"dgrad": "int8", "wgrad": "bf16"}


def test_quant_rejects_a_gemm_that_does_not_consume_the_tensor():
    with pytest.raises(ValueError):
        QuantizationConfig(enabled=True, dtype={"weight": {"wgrad": "fp8_e4m3"}})


def test_quant_dtype_resolves_explicit_tensors():
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
    )
    assert q.dtype == {
        "weight": {"fwd": "fp8_e4m3", "dgrad": "fp8_e4m3"},
        "act": {"fwd": "fp8_e4m3", "wgrad": "fp8_e4m3"},
        "grad_out": {"dgrad": "fp8_e5m2", "wgrad": "fp8_e5m2"},
    }


SCOPED_GRAD_OUT_DTYPES = [
    {"dgrad": "fp8_e5m2", "wgrad": "bf16"},
    {"dgrad": "bf16", "wgrad": "fp8_e5m2"},
]


@pytest.mark.parametrize("grad_out_dtype", SCOPED_GRAD_OUT_DTYPES)
def test_quant_dtype_scoped_gemm_is_explicit(grad_out_dtype):
    q = QuantizationConfig(
        enabled=True,
        dtype={
            "weight": "fp8_e4m3",
            "act": "fp8_e4m3",
            "grad_out": copy.deepcopy(grad_out_dtype),
        },
    )
    assert q.dtype["grad_out"] == grad_out_dtype


def test_quant_include_defaults():
    q = QuantizationConfig()
    assert q.include == ["*"] and q.exclude == ["lm_head", "*mlp.router.gate"]
    q2 = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        include=["*.mlp.*"],
    )
    assert q2.include == ["*.mlp.*"]


QUANTIZATION_SELECTORS = ["include", "exclude"]
INVALID_SELECTOR_PATTERNS = ["*.mlp.*", [1]]


@pytest.mark.parametrize("selector", QUANTIZATION_SELECTORS)
@pytest.mark.parametrize("patterns", INVALID_SELECTOR_PATTERNS)
def test_quantization_config_selectors_raise_error(selector, patterns):
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True, dtype={"weight": "fp8_e4m3"}, **{selector: patterns}
        )


QUANTIZATION_DICT_FIELDS = ["scale", "rounding", "rotation"]


@pytest.mark.parametrize("field", QUANTIZATION_DICT_FIELDS)
def test_quantization_config_disabled_fields_raise_error(field):
    with pytest.raises(ValueError):
        QuantizationConfig(**{field: "invalid"})


def test_quant_rejects_unknown_format():
    with pytest.raises(ValueError):
        QuantizationConfig(enabled=True, dtype={"weight": "not_a_fmt"})


ROW_SCALE_CASES = [
    (
        {"weight": {"granularity": "rowwise"}},
        {"weight": "rowwise", "act": "tensorwise", "grad_out": "tensorwise"},
    ),
    (
        {
            "weight": {"granularity": "rowwise"},
            "act": {"granularity": "rowwise"},
            "grad_out": {"granularity": "rowwise"},
        },
        {"weight": "rowwise", "act": "rowwise", "grad_out": "rowwise"},
    ),
]


@pytest.mark.parametrize("scale_and_expected", ROW_SCALE_CASES)
def test_quant_rowwise_scale(scale_and_expected):
    scale, expected_granularity = scale_and_expected
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        scale=scale,
    )
    assert {tensor: q.scale[tensor]["granularity"] for tensor in SCALE_TENSORS} == (
        expected_granularity
    )


@pytest.mark.parametrize(
    "dtype",
    [
        {"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        {"weight": "fp4_e2m1", "act": "fp4_e2m1", "grad_out": "fp4_e2m1"},
        {"weight": "int8", "act": "bf16", "grad_out": "bf16"},
    ],
)
def test_quant_default_scale_is_independent_of_dtype(dtype):
    q = QuantizationConfig(
        enabled=True,
        dtype=dtype,
    )
    assert q.scale == _scale_config(scale_dtype="fp32", enable_global_scale=False)


@pytest.mark.parametrize("granularity", ["tensorwise", "rowwise"])
@pytest.mark.parametrize("block_shape", OFF_BLOCKWISE_SHAPES)
def test_quant_block_shape_normalized_off_blockwise(granularity, block_shape):
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        scale={
            tensor: {
                "granularity": granularity,
                "block_shape": (
                    block_shape[tensor]
                    if isinstance(block_shape, dict)
                    else block_shape
                ),
            }
            for tensor in SCALE_TENSORS
        },
    )
    assert {tensor: q.scale[tensor]["block_shape"] for tensor in SCALE_TENSORS} == {
        tensor: (0, 0) if granularity == "tensorwise" else (1, 0)
        for tensor in SCALE_TENSORS
    }


INVALID_DTYPE_KEYS = ["bogus_grad", "recipe", "grad_input", "grad_weight", "dx", "dw"]


@pytest.mark.parametrize("dtype_key", INVALID_DTYPE_KEYS)
def test_quantization_config_dtype_key_raise_error(dtype_key):
    with pytest.raises(ValueError):
        QuantizationConfig(enabled=True, dtype={dtype_key: "bf16"})


def test_quant_rejects_unsupported_granularity():
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
            scale={"weight": {"granularity": "row"}},
        )


QUANTIZATION_LAYER_CASES = [
    (None, None),
    ([], []),
    ([0], [0]),
    ([11, 0, 2], [11, 0, 2]),
    ([-1, -2], []),
    ([2, -1, 0, 2, -3, 0], [2, 0]),
    ([12, 20], [12, 20]),
    ([12, 0, -1, 12], [12, 0]),
]


@pytest.mark.parametrize("case", QUANTIZATION_LAYER_CASES)
def test_train_config_quantization(tmp_path, case):
    layer_idx, expected_layers = case
    quantization_data = {
        "enabled": True,
        "enabled_after_steps": 2,
        "enabled_before_steps": 4,
        "layer_idx": layer_idx,
        "dtype": {"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
    }
    quantization = QuantizationConfig(**quantization_data)
    tc = TrainConfig(quantization=quantization)

    assert isinstance(tc.quantization, QuantizationConfig)
    assert tc.quantization.dtype["weight"]["fwd"] == "fp8_e4m3"
    assert tc.quantization.layer_idx == expected_layers
    assert tc.quantization is quantization

    exported = tc.to_dict()
    assert exported["quantization"]["enabled_after_steps"] == 2
    assert exported["quantization"]["enabled_before_steps"] == 4
    assert exported["quantization"]["layer_idx"] == expected_layers
    restored = load_config(_write_yaml(tmp_path, exported))
    assert restored.quantization.enabled_after_steps == 2
    assert restored.quantization.enabled_before_steps == 4
    assert restored.quantization.layer_idx == expected_layers


@pytest.mark.parametrize("explicit_config", [False, True])
def test_quant_disabled_stays_inert(explicit_config):
    config = (
        TrainConfig(quantization=QuantizationConfig())
        if explicit_config
        else TrainConfig()
    )
    quantization = config.quantization
    assert quantization.enabled is False
    assert quantization.dtype == {}
    assert quantization.scale == {}
    assert quantization.layer_idx is None
    assert quantization.rotation == {
        "rotation_cls": None,
        "rotation_kwargs": {},
        "rotation_axes": {},
    }


@pytest.mark.parametrize("enabled_after_steps", [-1, 1.0, "1", True])
def test_quantization_config_enabled_after_steps_raise_error(enabled_after_steps):
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            enabled_after_steps=enabled_after_steps,
        )


@pytest.mark.parametrize("enabled_before_steps", [-1, 1.0, "1", True])
def test_quantization_config_enabled_before_steps_raise_error(enabled_before_steps):
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            enabled_before_steps=enabled_before_steps,
        )


def test_quantization_config_enabled_steps():
    assert QuantizationConfig().enabled_after_steps == 0
    assert QuantizationConfig().enabled_before_steps is None
    assert (
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            enabled_before_steps=0,
        ).enabled_before_steps
        == 0
    )


# ---- mxfp8 / blockwise scaling (Option B: element format ⟂ scale scheme) ----


def test_quant_explicit_e8m0_config():
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e4m3"},
        scale=_scale_config(
            "blockwise",
            {tensor: (1, 32) for tensor in SCALE_TENSORS},
            scale_dtype="fp8_e8m0",
        ),
    )
    assert q.dtype == {
        "weight": {"fwd": "fp8_e4m3", "dgrad": "fp8_e4m3"},
        "act": {"fwd": "fp8_e4m3", "wgrad": "fp8_e4m3"},
        "grad_out": {"dgrad": "fp8_e4m3", "wgrad": "fp8_e4m3"},
    }
    assert q.scale == _scale_config(
        "blockwise",
        {tensor: (1, 32) for tensor in SCALE_TENSORS},
        scale_dtype="fp8_e8m0",
        enable_global_scale=False,
    )


def test_quant_explicit_nvfp4_scale():
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp4_e2m1", "act": "fp4_e2m1", "grad_out": "fp4_e2m1"},
        scale=_scale_config(
            "blockwise",
            {"weight": (16, 16), "act": (1, 16), "grad_out": (1, 16)},
            scale_dtype="fp8_e4m3",
            enable_global_scale=True,
        ),
    )
    assert {tensor: q.scale[tensor]["granularity"] for tensor in SCALE_TENSORS} == {
        tensor: "blockwise" for tensor in SCALE_TENSORS
    }
    assert {tensor: q.scale[tensor]["block_shape"] for tensor in SCALE_TENSORS} == {
        "weight": (16, 16),
        "act": (1, 16),
        "grad_out": (1, 16),
    }
    assert q.scale["scale_dtype"] == "fp8_e4m3"
    assert q.scale["enable_global_scale"] is True


QUANT_BLOCK_SHAPE_ERROR_SCALES = [
    {"weight": {"granularity": "blockwise", "block_shape": 16}},
    {"weight": {"granularity": "blockwise", "block_shape": None}},
    {"weight": {"granularity": "blockwise"}},
    {
        "weight": {"granularity": "blockwise", "block_shape": (1, 16, 16)},
    },
    {"weight": {"granularity": "blockwise", "block_shape": (1, 24)}},
    {"weight": {"granularity": "blockwise", "block_shape": (1, -16)}},
    {"weight": {"granularity": "blockwise", "block_shape": (16, 128)}},
]


@pytest.mark.parametrize("scale", QUANT_BLOCK_SHAPE_ERROR_SCALES)
def test_quant_block_shape_raise_error(scale):
    with pytest.raises(ValueError):
        QuantizationConfig(enabled=True, dtype={"weight": "fp8_e4m3"}, scale=scale)


UNKNOWN_SCALE_CONFIGS = [
    {"granularity": "rowwise"},
    {"block_shape": {"weight": (1, 16)}},
    {"recipe": "tensorwise"},
    {"unknown": "value"},
    {"weight": {"granularity": "rowwise", "unknown": "value"}},
    {"act": {"granularity": "rowwise", "unknown": "value"}},
    {"grad_out": {"granularity": "rowwise", "unknown": "value"}},
]


@pytest.mark.parametrize("unknown_scale", UNKNOWN_SCALE_CONFIGS)
def test_quant_ignores_unknown_scale_keys(unknown_scale):
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        scale={
            "weight": {"granularity": "rowwise"},
            "act": {"granularity": "rowwise"},
            "grad_out": {"granularity": "rowwise"},
            **unknown_scale,
        },
    )
    assert q.scale == _scale_config(
        "rowwise", scale_dtype="fp32", enable_global_scale=False
    )


def test_quant_scale_mixed_granularities_roundtrip():
    config = TrainConfig(
        training=TrainingConfig(mixed_precision="no"),
        quantization=QuantizationConfig(
            enabled=True,
            dtype={
                "weight": "fp8_e4m3",
                "act": "fp8_e4m3",
                "grad_out": "fp8_e5m2",
            },
            scale={
                "weight": {"granularity": "blockwise", "block_shape": (1, 32)},
                "act": {"granularity": "rowwise"},
            },
        ),
    )
    exported = config.to_dict()
    restored = TrainConfig(
        training=TrainingConfig(**exported["training"]),
        quantization=QuantizationConfig(**exported["quantization"]),
    )
    assert restored.quantization.scale == _scale_config(
        "tensorwise",
        scale_dtype="fp32",
        enable_global_scale=False,
    ) | {
        "weight": {"granularity": "blockwise", "block_shape": (1, 32)},
        "act": {"granularity": "rowwise", "block_shape": (1, 0)},
    }


# Full arithmetic scale dtypes absorb a global factor, leaving it nothing to do.
GLOBAL_SCALE_DEGENERATE_SCALES = [
    _scale_config(
        granularity,
        {tensor: (1, 16) for tensor in SCALE_TENSORS}
        if granularity == "blockwise"
        else None,
        scale_dtype="fp32",
    )
    for granularity in ("tensorwise", "rowwise", "blockwise")
] + [{}, {"scale_dtype": None}]


@pytest.mark.parametrize("scale", GLOBAL_SCALE_DEGENERATE_SCALES)
def test_quant_global_scale_disabled_for_wide_scale_dtype(scale):
    """Scale configuration normalizes global scaling off for fp32."""
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3"},
        scale={**scale, "enable_global_scale": True},
    )
    assert q.scale["enable_global_scale"] is False
    assert q.scale["scale_dtype"] == "fp32"


def test_quant_global_scale_kept_for_narrow_scale_dtype():
    """e4m3 block scales have a range to normalise into, at any granularity."""
    for granularity, extra in (
        ("tensorwise", {}),
        ("rowwise", {}),
        ("blockwise", {tensor: (1, 16) for tensor in SCALE_TENSORS}),
    ):
        q = QuantizationConfig(
            enabled=True,
            dtype={"weight": "int4", "act": "int4"},
            scale=_scale_config(
                granularity,
                extra if granularity == "blockwise" else None,
                scale_dtype="fp8_e4m3",
                enable_global_scale=True,
            ),
        )
        assert q.scale["enable_global_scale"] is True, granularity


def test_quant_global_scale_disabled_for_e8m0_scale_dtype():
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3"},
        scale=_scale_config(
            "blockwise",
            {tensor: (1, 32) for tensor in SCALE_TENSORS},
            scale_dtype="fp8_e8m0",
            enable_global_scale=True,
        ),
    )
    assert q.scale["enable_global_scale"] is False


def test_quant_global_scale_raise_error():
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "int4"},
            scale={"weight": {"granularity": "rowwise"}, "enable_global_scale": "yes"},
        )


def test_quant_explicit_scale_input_is_not_mutated():
    scale = _scale_config(
        "blockwise",
        {tensor: (1, 64) for tensor in SCALE_TENSORS},
        scale_dtype="fp8_e8m0",
    )
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e4m3"},
        scale=scale,
    )
    assert {tensor: q.scale[tensor]["block_shape"] for tensor in SCALE_TENSORS} == {
        "weight": (1, 64),
        "act": (1, 64),
        "grad_out": (1, 64),
    }
    assert scale == _scale_config(
        "blockwise",
        {tensor: (1, 64) for tensor in SCALE_TENSORS},
        scale_dtype="fp8_e8m0",
    )


def test_quant_e8m0_requires_blockwise():
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            scale={"weight": {"granularity": "rowwise"}, "scale_dtype": "fp8_e8m0"},
        )


def test_quant_e8m0_rejects_int_element():
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "int8", "act": "fp8_e4m3", "grad_out": "fp8_e4m3"},
            scale=_scale_config(
                "blockwise",
                {tensor: (1, 32) for tensor in SCALE_TENSORS},
                scale_dtype="fp8_e8m0",
            ),
        )


@pytest.mark.parametrize("scale", [None, []])
def test_quant_scale_raise_error(scale):
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e4m3"},
            scale=scale,
        )


@pytest.mark.parametrize("scale_dtype", ({}, {"scale_dtype": None}))
def test_quant_rowwise_defaults_scale_dtype_to_fp32(scale_dtype):
    # adding blockwise validation must not disturb the existing rowwise path
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3"},
        scale={"weight": {"granularity": "rowwise"}, **scale_dtype},
    )
    assert q.scale["weight"]["granularity"] == "rowwise"
    assert q.scale["scale_dtype"] == "fp32"


# ==================== latent_moe / latent_dim ====================


@pytest.mark.parametrize("latent_dim", [None, 0, -4, 3.5])
def test_modelconfig_latent_moe_requires_positive_latent_dim(latent_dim):
    kwargs = {"n_routed_experts": 4, "aux_loss": True, "latent_moe": True}
    if latent_dim is not None:
        kwargs["latent_dim"] = latent_dim
    with pytest.raises(ValueError):
        ModelConfig(d_model=64, mlp=[{"mlp_cls": "moe", "mlp_kwargs": kwargs}])


def test_modelconfig_latent_moe_valid():
    cfg = ModelConfig(
        d_model=64,
        mlp=[
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {
                    "n_routed_experts": 4,
                    "aux_loss": True,
                    "latent_moe": True,
                    "latent_dim": 16,
                },
            }
        ],
    )
    assert cfg.resolve_mlp(0)[1]["latent_moe"] is True
    assert cfg.resolve_mlp(0)[1]["latent_dim"] == 16


# ==================== per-layer mlp schema + resolver ====================


def _moe_kwargs(expert_bias_update_rate=None):
    kwargs = {"n_routed_experts": 4, "expert_bias": True}
    if expert_bias_update_rate is not None:
        kwargs["expert_bias_update_rate"] = expert_bias_update_rate
    return kwargs


def test_mlp_single_item_covers_all_layers():
    cfg = ModelConfig(
        d_model=64, n_layers=4, mlp=[{"mlp_cls": "moe", "mlp_kwargs": _moe_kwargs()}]
    )
    assert [cfg.resolve_mlp(i)[0] for i in range(cfg.n_layers)] == ["moe"] * 4
    assert cfg.is_moe is True


def test_mlp_dense_first_layer_complement():
    cfg = ModelConfig(
        d_model=64,
        n_layers=4,
        mlp=[
            {"mlp_cls": "dense", "mlp_kwargs": {}, "layer_idx": [0]},
            {"mlp_cls": "moe", "mlp_kwargs": _moe_kwargs()},  # complement -> [1,2,3]
        ],
    )
    assert [cfg.resolve_mlp(i)[0] for i in range(cfg.n_layers)] == [
        "dense",
        "moe",
        "moe",
        "moe",
    ]
    assert cfg.mlp[1]["layer_idx"] == [1, 2, 3]


def test_mlp_per_layer_expert_bias_rate():
    cfg = ModelConfig(
        d_model=64,
        n_layers=3,
        mlp=[
            {
                "mlp_cls": "moe",
                "mlp_kwargs": _moe_kwargs(expert_bias_update_rate=0.004),
                "layer_idx": [0],
            },
            {"mlp_cls": "moe", "mlp_kwargs": _moe_kwargs()},  # rate defaults to 0.001
        ],
    )
    assert cfg.resolve_mlp(0)[1]["expert_bias_update_rate"] == 0.004
    assert cfg.resolve_mlp(1)[1]["expert_bias_update_rate"] == 0.001


def test_mlp_conflict_raises():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=4,
            mlp=[
                {"mlp_cls": "dense", "layer_idx": [0]},
                {"mlp_cls": "moe", "mlp_kwargs": _moe_kwargs(), "layer_idx": [0, 1]},
                {"mlp_cls": "dense"},
            ],
        )


def test_mlp_duplicate_layer_raise_error():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=4,
            mlp=[
                {"mlp_cls": "dense", "layer_idx": [0, 0]},
                {"mlp_cls": "dense"},
            ],
        )


def test_mlp_gap_raises():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=4,
            mlp=[
                {"mlp_cls": "dense", "layer_idx": [0]},
                {"mlp_cls": "moe", "mlp_kwargs": _moe_kwargs(), "layer_idx": [1]},
            ],
        )


def test_mlp_two_bare_items_raise():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=4,
            mlp=[
                {"mlp_cls": "dense"},
                {"mlp_cls": "dense"},
            ],
        )


def test_mlp_out_of_range_raises():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=64,
            n_layers=2,
            mlp=[
                {"mlp_cls": "dense", "layer_idx": [5]},
                {"mlp_cls": "dense"},
            ],
        )


def test_mlp_per_layer_aux_coef_allowed():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        mlp=[
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {
                    "n_routed_experts": 4,
                    "aux_loss": True,
                    "aux_loss_coef": 1e-3,
                },
                "layer_idx": [0],
            },
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {
                    "n_routed_experts": 4,
                    "aux_loss": True,
                    "aux_loss_coef": 1e-2,
                },
            },
        ],
    )
    coefs = [
        cfg.resolve_mlp(i)[1]["aux_loss_coef"]
        for i in range(cfg.n_layers)
        if cfg.resolve_mlp(i)[0] == "moe" and cfg.resolve_mlp(i)[1].get("aux_loss")
    ]
    assert sorted(coefs) == [1e-3, 1e-2]


def test_all_configs_load_and_have_list_mlp():
    paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
        "experiments/**/*.yaml", recursive=True
    )
    assert paths
    for p in paths:
        cfg = load_config(p)
        assert isinstance(cfg.model.mlp, list) and cfg.model.mlp
        # resolver ran and covers every layer
        assert all(cfg.model.resolve_mlp(i)[0] for i in range(cfg.model.n_layers))
        assert isinstance(cfg.model.attn, list) and cfg.model.attn
        assert all(cfg.model.resolve_attn(i)[0] for i in range(cfg.model.n_layers))


def test_configs_model_key_order_d_model_n_layers_vocab_size_attn_mlp_first():
    for p in ("configs/gpt2_124m.yaml", "configs/qwen3_51m.yaml"):
        raw = yaml.safe_load(open(p))
        keys = list(raw["model"].keys())
        assert keys[:3] == ["d_model", "n_layers", "vocab_size"], (p, keys)
        assert keys[3] == "attn" and keys[4] == "mlp", (p, keys)


# ==================== Scaling configuration ====================


ROTATION_DEFAULT_EXTRAS = [
    {},
    {"rotation_ops": {}, "gemms": ["fwd"]},
]


@pytest.mark.parametrize("enabled", QUANTIZATION_ENABLED)
@pytest.mark.parametrize("extra", ROTATION_DEFAULT_EXTRAS)
def test_quantization_config_rotation_defaults(extra, enabled):
    """Config canonicalization applies defaults without storing runtime state."""
    rotation = {"rotation_cls": "hadamard"}
    rotation.update(extra)
    q = QuantizationConfig(
        enabled=enabled,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        rotation=rotation,
    )
    assert q.rotation["rotation_cls"] == "hadamard"
    assert q.rotation["rotation_kwargs"] == {}
    expected_axes = (
        {
            "weight": {"fwd": [], "dgrad": []},
            "act": {"fwd": [], "wgrad": []},
            "grad_out": {"dgrad": [], "wgrad": []},
        }
        if enabled
        else {}
    )
    assert q.rotation["rotation_axes"] == expected_axes
    exported = TrainConfig(quantization=q).to_dict()
    assert exported["quantization"]["rotation"] == q.rotation
    yaml.safe_dump(exported)


@pytest.mark.parametrize(
    "rotation",
    [
        {},
        {"rotation_cls": None},
        {"rotation_kwargs": {"block_size": 32}},
        {"rotation_axes": {"act": {"fwd": [-2]}}},
    ],
)
def test_quantization_config_rotation_requires_rotation_cls(rotation):
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        rotation=rotation,
    )
    assert q.rotation == {
        "rotation_cls": None,
        "rotation_kwargs": {},
        "rotation_axes": {
            "weight": {"fwd": [], "dgrad": []},
            "act": {"fwd": [], "wgrad": []},
            "grad_out": {"dgrad": [], "wgrad": []},
        },
    }


ROTATION_KWARGS = [
    {"block_size": 1},
    {"block_size": 2},
    {"block_size": 32},
    {"block_size": 128},
    {},
    {"seed": 7},
]


@pytest.mark.parametrize("rotation_kwargs", ROTATION_KWARGS)
def test_quantization_config_rotation_kwargs(rotation_kwargs):
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        rotation={
            "rotation_cls": "hadamard",
            "rotation_kwargs": copy.deepcopy(rotation_kwargs),
        },
    )
    assert q.rotation["rotation_cls"] == "hadamard"
    assert q.rotation["rotation_kwargs"] == rotation_kwargs
    config = TrainConfig(seed=23, quantization=q)
    assert config.quantization.rotation["rotation_kwargs"] == rotation_kwargs


@pytest.mark.parametrize("enabled", QUANTIZATION_ENABLED)
def test_quantization_config_rotation_axes(enabled):
    rotation_axes = {
        "weight": {"fwd": [-1, -2, -1], "dgrad": [-2]},
        "act": {"wgrad": [-1]},
        "grad_out": {"dgrad": [-2]},
    }
    q = QuantizationConfig(
        enabled=enabled,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        rotation={
            "rotation_cls": "hadamard",
            "rotation_axes": rotation_axes,
        },
    )
    expected_axes = (
        {
            "weight": {"fwd": [-2, -1], "dgrad": [-2]},
            "act": {"fwd": [], "wgrad": [-1]},
            "grad_out": {"dgrad": [-2], "wgrad": []},
        }
        if enabled
        else rotation_axes
    )
    assert q.rotation["rotation_axes"] == expected_axes


@pytest.mark.parametrize(
    "rotation",
    [
        None,
        [],
        {"rotation_cls": []},
        {"rotation_cls": "givens"},
        {"rotation_cls": "hadamard", "rotation_kwargs": 4},
        {"rotation_cls": "hadamard", "rotation_kwargs": []},
        {"rotation_cls": "hadamard", "rotation_axes": []},
        {"rotation_cls": "hadamard", "rotation_axes": {"output": {}}},
        {"rotation_cls": "hadamard", "rotation_axes": {"weight": []}},
        {
            "rotation_cls": "hadamard",
            "rotation_axes": {"weight": {"wgrad": []}},
        },
        {
            "rotation_cls": "hadamard",
            "rotation_axes": {"weight": {"fwd": "-1"}},
        },
        {
            "rotation_cls": "hadamard",
            "rotation_axes": {"weight": {"fwd": [-3]}},
        },
        {
            "rotation_cls": "hadamard",
            "rotation_axes": {"weight": {"fwd": [True]}},
        },
    ],
)
def test_quantization_config_rotation_raise_error(rotation):
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
            rotation=rotation,
        )


def test_quantization_config_rotation_kwargs_raise_error():
    """Kwargs must survive the YAML round trip that carries them into a run."""
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
            rotation={
                "rotation_cls": "hadamard",
                "rotation_kwargs": {"sign_vector": torch.ones(4)},
            },
        )


BLOCKWISE_SHAPES = [(1, 128), (16, 16), (32, 32), (64, 64)]


@pytest.mark.parametrize("block_shape", BLOCKWISE_SHAPES)
def test_quantization_config_blockwise(block_shape):
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
        scale=_scale_config(
            "blockwise", {tensor: block_shape for tensor in SCALE_TENSORS}
        ),
    )
    assert {tensor: q.scale[tensor]["granularity"] for tensor in SCALE_TENSORS} == {
        tensor: "blockwise" for tensor in SCALE_TENSORS
    }
    assert {tensor: q.scale[tensor]["block_shape"] for tensor in SCALE_TENSORS} == {
        tensor: block_shape for tensor in SCALE_TENSORS
    }
    assert q.scale["scale_dtype"] == "fp32"


def test_quant_rejects_unknown_scale_dtype():
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            scale=_scale_config(
                "blockwise",
                {tensor: (1, 32) for tensor in SCALE_TENSORS},
                scale_dtype="e3m4",
            ),
        )


def test_quant_blockwise_fp32_scale_dtype_ok():
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3"},
        scale=_scale_config(
            "blockwise",
            {tensor: (1, 64) for tensor in SCALE_TENSORS},
            scale_dtype="fp32",
        ),
    )
    assert q.scale["scale_dtype"] == "fp32"


def test_monitoring_flags_default_true():
    from src.utils.config import LoggingConfig

    assert LoggingConfig().log_quant_metrics is True
    assert LoggingConfig().log_activation_norms is True


def test_quant_e8m0_requires_contract_extent_multiple_of_32():
    with pytest.raises(ValueError):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e4m3"},
            scale=_scale_config(
                "blockwise",
                {tensor: (16, 16) for tensor in SCALE_TENSORS},
                scale_dtype="fp8_e8m0",
            ),
        )
