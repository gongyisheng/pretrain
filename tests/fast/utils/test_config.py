import copy
import os
import glob
import json
import tempfile
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


def _write_yaml(tmp_dir, data):
    path = os.path.join(tmp_dir, "test.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


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


def test_load_config_from_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        assert config.max_seq_len == 128
        assert config.model.n_layers == 2
        assert config.optimizer.lr == 1e-3


def test_load_config_model_kwargs():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        cfg = load_config(path)
        assert cfg.model.resolve_attn(0)[1]["n_heads"] == 4
        assert cfg.model.resolve_mlp(0)[1]["activation_cls"] == "swiglu"
        assert cfg.model.pos_emb_kwargs["rope_theta"] == 1e4


def test_config_to_dict_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        d = config.to_dict()
        assert d["max_seq_len"] == 128
        assert d["model"]["attn"][0]["attn_cls"] == "gqa"
        assert d["model"]["attn"][0]["attn_kwargs"]["n_heads"] == 4


@pytest.mark.parametrize(
    ("scale", "scale_dtype"),
    [
        ({"granularity": "rowwise", "scale_dtype": "fp32"}, "fp32"),
        (
            {
                "granularity": "blockwise",
                "block_shape": [1, 32],
                "scale_dtype": "fp8_e8m0",
            },
            "fp8_e8m0",
        ),
    ],
)
def test_config_to_dict_serializes_quant_scale_dtype(scale, scale_dtype):
    config = TrainConfig(
        training=TrainingConfig(
            mixed_precision="no",
            quantization={
                "enabled": True,
                "dtype": {"weight": "fp8_e4m3"},
                "scale": scale,
            },
        )
    )

    exported = config.to_dict()

    assert (
        exported["training"]["quantization"][0]["scale"]["scale_dtype"] == scale_dtype
    )
    json_export = json.dumps(exported)
    yaml_export = yaml.safe_dump(exported)
    assert (
        json.loads(json_export)["training"]["quantization"][0]["scale"]["scale_dtype"]
        == scale_dtype
    )
    assert (
        yaml.safe_load(yaml_export)["training"]["quantization"][0]["scale"][
            "scale_dtype"
        ]
        == scale_dtype
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        with open(path, "w") as f:
            f.write(yaml_export)
        restored = load_config(path)
    assert _only_rule(restored.training).scale["scale_dtype"] is getattr(
        torch, "float32" if scale_dtype == "fp32" else "float8_e8m0fnu"
    )


@pytest.mark.parametrize(
    ("runtime_dtype", "scale_dtype"),
    [
        (torch.float32, "fp32"),
        (torch.float8_e8m0fnu, "fp8_e8m0"),
    ],
)
def test_config_to_dict_serializes_disabled_quant_scale_dtype(
    runtime_dtype, scale_dtype
):
    config = TrainConfig(
        training=TrainingConfig(
            quantization={
                "enabled": False,
                "scale": {"scale_dtype": runtime_dtype},
            },
        )
    )

    exported = config.to_dict()

    assert (
        exported["training"]["quantization"][0]["scale"]["scale_dtype"] == scale_dtype
    )
    json.dumps(exported)
    yaml.safe_dump(exported)


def test_config_to_dict_rotation_roundtrip():
    config = TrainConfig(
        training=TrainingConfig(
            mixed_precision="no",
            quantization={
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "rotation": {
                    "rotation_cls": "hadamard",
                    "rotation_kwargs": {
                        "block_size": 4,
                        "random_sign": True,
                        "sign_vector": [1.0, -1.0, -1.0, 1.0],
                    },
                    "gemms": ["fwd", "wgrad"],
                },
            },
        )
    )
    original_rule = _only_rule(config.training)
    assert not hasattr(original_rule, "_rotation")

    exported = config.to_dict()
    exported_rotation = exported["training"]["quantization"][0]["rotation"]
    assert set(exported_rotation) == {"rotation_cls", "rotation_kwargs", "gemms"}
    assert exported_rotation["rotation_kwargs"]["sign_vector"] == [
        1.0,
        -1.0,
        -1.0,
        1.0,
    ]
    json.loads(json.dumps(exported))
    yaml_export = yaml.safe_dump(exported)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        with open(path, "w") as f:
            f.write(yaml_export)
        restored = load_config(path)

    restored_rule = _only_rule(restored.training)
    assert restored_rule.rotation == original_rule.rotation


# ==================== CLI overrides ====================


def test_load_config_optimizer_section():
    with tempfile.TemporaryDirectory() as tmp:
        # The section is optional (tokenizer-training configs have none) and
        # falls back to the dataclass defaults.
        raw = copy.deepcopy(MINIMAL_CONFIG)
        del raw["optimizer"]
        cfg = load_config(_write_yaml(tmp, raw)).optimizer
        assert (cfg.optimizer_cls, cfg.lr) == ("adamw", 5e-4)
        assert cfg.optimizer_kwargs == {
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "fused": True,
        }


def test_config_cli_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(
            path, overrides=["optimizer.lr=3e-4", "training.batch_size=8"]
        )
        assert config.optimizer.lr == 3e-4
        assert config.training.batch_size == 8


def test_config_cli_override_nested_kwargs():
    # attn/mlp kwargs live inside per-layer list items now (not directly
    # overridable by dotted path); exercise the same nested-dict-override
    # mechanism against pos_emb_kwargs, which is still a flat dict field.
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        cfg = load_config(path, overrides=["model.pos_emb_kwargs.rope_theta=5000"])
        assert cfg.model.pos_emb_kwargs["rope_theta"] == 5000


# ==================== Nested coercion ====================


def test_load_config_coerces_nested_kwargs(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  pos_emb_kwargs:\n    rope_theta: 1e4\n")
    cfg = load_config(str(p))
    assert cfg.model.pos_emb_kwargs["rope_theta"] == 10000.0


# ==================== TrainingConfig / DataConfig ====================


def test_training_config_intra_doc_masking_default():
    cfg = TrainingConfig()
    assert cfg.intra_doc_masking is True


def test_data_config_packing_default():
    cfg = DataConfig()
    assert cfg.packing is True


def test_intra_doc_masking_yaml_override(tmp_path):
    yaml_content = """
max_seq_len: 128
training:
  intra_doc_masking: false
data:
  packing: true
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.training.intra_doc_masking is False
    assert cfg.data.packing is True


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


def test_default_task_is_pretrain():
    from src.utils.config import TrainConfig

    cfg = TrainConfig()
    assert cfg.task == "pretrain"


def test_task_accepts_sft():
    from src.utils.config import TrainConfig

    cfg = TrainConfig()
    cfg.task = "sft"
    assert cfg.task == "sft"


def test_default_eval_train_is_false():
    from src.utils.config import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.eval_train is False


# ==================== attn ====================


def test_attn_cls_defaults():
    # Default attn_cls is gqa
    assert ModelConfig().resolve_attn(0)[0] == "gqa"
    assert (
        ModelConfig(attn=[{"attn_cls": "mha", "attn_kwargs": {}}]).resolve_attn(0)[0]
        == "mha"
    )
    assert (
        ModelConfig(attn=[{"attn_cls": "mla", "attn_kwargs": {}}]).resolve_attn(0)[0]
        == "mla"
    )


def test_attn_kwargs_preserved():
    # attn_implementation default is filled; explicit values are preserved
    assert ModelConfig().resolve_attn(0)[1] == {"attn_implementation": "flex_attention"}
    assert (
        ModelConfig(
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 8}}]
        ).resolve_attn(0)[1]["n_heads"]
        == 8
    )


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
            {"attn_cls": "mha", "attn_kwargs": {"n_heads": 4}, "layer_idx": [0]},
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


def test_attn_conflict_raises():
    with pytest.raises(ValueError, match="claimed by multiple"):
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
    with pytest.raises(ValueError, match="no attn item|coverage"):
        ModelConfig(
            d_model=64,
            n_layers=4,
            attn=[
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}, "layer_idx": [0]},
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}, "layer_idx": [1]},
            ],
        )


def test_attn_two_bare_items_raise():
    with pytest.raises(ValueError, match="at most one"):
        ModelConfig(
            d_model=64,
            n_layers=4,
            attn=[
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}},
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}},
            ],
        )


def test_attn_dup_index_within_item_raises():
    with pytest.raises(ValueError, match="more than once in one attn item"):
        ModelConfig(
            d_model=64,
            n_layers=4,
            attn=[
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}, "layer_idx": [0, 0]},
                {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4}},
            ],
        )


def test_attn_defaulting_matches_mla_and_gqa():
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


def test_attn_n_heads_divisibility_raises():
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ModelConfig(
            d_model=64,
            n_layers=2,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 5}}],
        )


def test_attn_implementation_must_be_shared_across_layers():
    # The trainer builds one attention mask shared across layers, so layers
    # cannot disagree on attn_implementation.
    with pytest.raises(ValueError, match="attn_implementation"):
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


def test_modelconfig_moe_expert_bias_defaults():
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
    with pytest.raises(ValueError, match="both are on"):
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
    with pytest.raises(ValueError, match="both are off"):
        ModelConfig(
            d_model=64,
            mlp=[{"mlp_cls": "moe", "mlp_kwargs": {"n_routed_experts": 4}}],
        )


def test_modelconfig_moe_router_score_fn_defaults_sigmoid():
    cfg = ModelConfig(
        d_model=64,
        mlp=[
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {"n_routed_experts": 4, "aux_loss": True},
            }
        ],
    )
    assert cfg.resolve_mlp(0)[1]["router_score_fn"] == "sigmoid"


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
    with pytest.raises(ValueError, match="unknown router_score_fn"):
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
    # (activation_cls, activation_kwargs, expected message) -- an inverted bound is
    # the only act_limit defect that raises; the rest are tolerated below
    (
        "swiglu",
        {"act_limit": {"up": {"min": 7, "max": 1}}},
        r"act_limit\['up'\] requires min < max",
    ),
    (
        "swiglu",
        {"act_limit": {"up": {"min": 0, "max": -1}}},  # 0 is a real bound
        r"act_limit\['up'\] requires min < max",
    ),
    (
        "silu",
        {"act_limit": {"up": {"min": 7, "max": 1}}},
        r"act_limit\['up'\] requires min < max",
    ),
]

# (activation_cls, act_limit) shapes the check does not police -- kept verbatim and
# left for the activation, which only reads its own 'gate'/'up' numeric bounds
ACT_LIMITS_TOLERATED = [
    ("swiglu", {"max": 7}),  # side keys that name no tensor
    ("silu", {"gate": {"max": 7}}),  # gate limit on a unary activation
    ("swiglu", {"gate": 7}),  # bounds that are not a dict
    ("swiglu", {"gate": {}}),
    ("swiglu", {"gate": {"lo": 1}}),  # bound keys that are not min/max
    ("swiglu", {"up": {"min": "x"}}),  # non-numeric bound
]

ACT_LIMITS_OK = [
    ({"gate": {"max": 7}}, "swiglu"),  # partial: only the gate's upper tail
    ({"gate": {"max": 7}, "up": {"min": -7, "max": 7}}, "swiglu"),  # deepseek-v4
    ({"up": {"min": -7}}, "swiglu"),
    ({"up": {"max": 7}}, "silu"),  # unary clamps its single up tensor
    ({"up": {"min": -7, "max": 7}}, "silu"),
]


def test_modelconfig_activation_cls_raise_error():
    with pytest.raises(ValueError, match="unknown activation_cls"):
        ModelConfig(
            mlp=[{"mlp_cls": "dense", "mlp_kwargs": {"activation_cls": "mish"}}]
        )


@pytest.mark.parametrize("activation_cls", ["swiglu", "silu", "bilinear", "powlu"])
def test_modelconfig_activation_cls(activation_cls):
    """The name alone resolves the activation; SwiGLU is the default."""
    resolved = ModelConfig(d_model=64).resolve_mlp(0)[1]
    assert resolved["activation_cls"] == "swiglu"
    assert resolved["activation_kwargs"] == {}
    cfg = ModelConfig(
        d_model=64,
        mlp=[{"mlp_cls": "dense", "mlp_kwargs": {"activation_cls": activation_cls}}],
    )
    assert cfg.resolve_mlp(0)[1]["activation_cls"] == activation_cls


@pytest.mark.parametrize("mlp_cls", ["dense", "moe"])
@pytest.mark.parametrize("act_cls,act_kwargs,match", ACT_KWARGS_ERRORS)
def test_modelconfig_activation_kwargs_raise_error(mlp_cls, act_cls, act_kwargs, match):
    kwargs = {"activation_cls": act_cls, "activation_kwargs": act_kwargs}
    if mlp_cls == "moe":
        kwargs |= {"n_routed_experts": 4, "aux_loss": True}
    with pytest.raises(ValueError, match=match):
        ModelConfig(d_model=64, mlp=[{"mlp_cls": mlp_cls, "mlp_kwargs": kwargs}])


@pytest.mark.parametrize("act_kwargs", [7.0, "off", None])
def test_modelconfig_activation_kwargs_non_mapping_dropped(act_kwargs):
    """A non-dict activation_kwargs is discarded, leaving the block's own default."""
    cfg = ModelConfig(
        d_model=64,
        mlp=[{"mlp_cls": "dense", "mlp_kwargs": {"activation_kwargs": act_kwargs}}],
    )
    assert "activation_kwargs" not in cfg.resolve_mlp(0)[1]


@pytest.mark.parametrize("act_cls,act_limit", ACT_LIMITS_TOLERATED)
def test_modelconfig_activation_kwargs_act_limit_tolerated(act_cls, act_limit):
    """Malformed-but-dict limits are stored verbatim rather than rejected."""
    cfg = ModelConfig(
        d_model=64,
        mlp=[
            {
                "mlp_cls": "dense",
                "mlp_kwargs": {
                    "activation_cls": act_cls,
                    "activation_kwargs": {"act_limit": act_limit},
                },
            }
        ],
    )
    assert cfg.resolve_mlp(0)[1]["activation_kwargs"] == {"act_limit": act_limit}


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


@pytest.mark.parametrize("act_limit,activation_cls", ACT_LIMITS_OK)
def test_modelconfig_activation_kwargs_act_limit(act_limit, activation_cls):
    """Valid limits are stored verbatim; the shape follows the activation's arity."""
    cfg = ModelConfig(
        d_model=64,
        mlp=[
            {
                "mlp_cls": "dense",
                "mlp_kwargs": {
                    "activation_cls": activation_cls,
                    "activation_kwargs": {"act_limit": act_limit},
                },
            }
        ],
    )
    assert cfg.resolve_mlp(0)[1]["activation_kwargs"] == {"act_limit": act_limit}


@pytest.mark.parametrize(
    "act_kwargs",
    [
        {"alpha": 1.702},
        {"up_shift": 1.0},
        # the gpt-oss SwiGLU, now expressible without a dedicated registry entry
        {
            "alpha": 1.702,
            "up_shift": 1.0,
            "act_limit": {"gate": {"max": 7.0}, "up": {"min": -7.0, "max": 7.0}},
        },
    ],
)
def test_modelconfig_activation_kwargs(act_kwargs):
    cfg = ModelConfig(
        d_model=64,
        mlp=[
            {
                "mlp_cls": "dense",
                "mlp_kwargs": {
                    "activation_cls": "swiglu",
                    "activation_kwargs": act_kwargs,
                },
            }
        ],
    )
    assert cfg.resolve_mlp(0)[1]["activation_kwargs"] == act_kwargs


def test_modelconfig_validates_attn_dims():
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ModelConfig(
            d_model=100, attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 3}}]
        )
    with pytest.raises(ValueError, match="divisible by\\s+n_kv_heads"):
        ModelConfig(
            d_model=64,
            attn=[
                {
                    "attn_cls": "gqa",
                    "attn_kwargs": {"n_heads": 4, "n_kv_heads": 3},
                }
            ],
        )


def test_modelconfig_gqa_defaults_n_kv_heads():
    cfg = ModelConfig(
        d_model=64, attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 8}}]
    )
    assert cfg.resolve_attn(0)[1]["n_kv_heads"] == 8  # defaults to n_heads


# ==================== string-field validation ====================


def test_scheduler_unknown_name_raises():
    from src.utils.config import SchedulerConfig

    with pytest.raises(ValueError, match="unknown scheduler"):
        SchedulerConfig(name="step")
    SchedulerConfig(name="cosine")
    SchedulerConfig(name="constant")


@pytest.mark.parametrize(
    "optimizer_cls, expected",
    [
        ("adamw", {"betas": (0.9, 0.95), "eps": 1e-8, "fused": True}),
        ("lion", {"betas": (0.9, 0.99), "foreach": True}),
        (
            "muon",
            {
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "momentum": 0.95,
                "nesterov": True,
                "adjust_lr_fn": "match_rms_adamw",
                "fused": True,
            },
        ),
    ],
)
def test_optimizer_config_defaults(optimizer_cls, expected):
    from src.utils.config import OptimizerConfig

    # Unset kwargs get the pretraining-tuned defaults for the selected optimizer.
    assert OptimizerConfig(optimizer_cls, lr=1e-3).optimizer_kwargs == expected
    # An explicit value wins; the remaining keys are still filled in.
    for key in expected:
        cfg = OptimizerConfig(optimizer_cls, lr=1e-3, optimizer_kwargs={key: "set"})
        assert cfg.optimizer_kwargs == {**expected, key: "set"}


def test_optimizer_config_raise_error():
    from src.utils.config import OptimizerConfig

    with pytest.raises(ValueError, match="unknown optimizer_cls"):
        OptimizerConfig("sgd", lr=1e-3)


def test_training_unknown_mixed_precision_raises():
    with pytest.raises(ValueError, match="unknown mixed_precision"):
        TrainingConfig(mixed_precision="fp8")


def test_training_unknown_loss_fn_raises():
    with pytest.raises(ValueError, match="unknown loss_fn"):
        TrainingConfig(loss_fn="huber")


def test_training_device_default_and_validation():
    assert TrainingConfig().device == "auto"
    assert TrainingConfig(device="cpu").device == "cpu"
    with pytest.raises(ValueError, match="unknown device"):
        TrainingConfig(device="tpu")


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
    with pytest.raises(ValueError, match="'bf16' or 'fp16'"):
        _moe_train_config("no")


@pytest.mark.parametrize("mixed_precision", ["bf16", "fp16"])
def test_dropless_moe_reduced_precision_ok(mixed_precision):
    cfg = _moe_train_config(mixed_precision)
    assert cfg.training.mixed_precision == mixed_precision


# ==================== Quantization config ====================


def _only_rule(tc):
    # TrainingConfig normalizes quant to a list of rules; single-rule helper.
    assert len(tc.quantization) == 1
    return tc.quantization[0]


def test_quant_defaults_disabled():
    q = QuantizationConfig()
    assert q.enabled is False
    assert q.rounding == {}
    assert q.exclude == ["lm_head", "*mlp.router.gate"]
    # disabled rule is inert: no dtype/scale defaults applied
    assert q.dtype == {} and q.scale == {}


def test_quant_rounding_defaults_to_rne_everywhere():
    q = _only_rule(TrainingConfig(quantization={"enabled": True}))
    assert q.rounding == {"weight": "RNE", "act": "RNE", "grad_out": "RNE"}
    assert not any(mode == "SR" for mode in q.rounding.values())


def test_quant_rounding_fills_unnamed_tensors_and_round_trips():
    config = TrainConfig(
        training=TrainingConfig(
            mixed_precision="bf16",
            quantization={"enabled": True, "rounding": {"grad_out": "SR"}},
        )
    )
    q = _only_rule(config.training)
    assert q.rounding == {"weight": "RNE", "act": "RNE", "grad_out": "SR"}
    assert config.to_dict()["training"]["quantization"][0]["rounding"] == q.rounding


def test_quant_rounding_on_an_unquantized_tensor_is_inert():
    # act stays at the compute dtype here; a mode on it names no quantizer, and the
    # config takes it rather than second-guessing a dtype it may be swept against.
    q = _only_rule(
        TrainingConfig(
            mixed_precision="bf16",
            quantization={
                "enabled": True,
                "dtype": {"weight": "int8"},
                "rounding": {"act": "SR"},
            },
        )
    )
    assert q.dtype["act"] == {"fwd": "bf16", "wgrad": "bf16"}
    assert q.rounding["act"] == "SR"


def test_quant_rounding_rejects_unknown_key_and_mode():
    with pytest.raises(ValueError, match="unknown quant rounding key: 'grad_input'"):
        QuantizationConfig(enabled=True, rounding={"grad_input": "SR"})
    with pytest.raises(
        ValueError, match="unknown quant rounding for act: 'stochastic'"
    ):
        QuantizationConfig(enabled=True, rounding={"act": "stochastic"})


def test_quant_tensor_defaults_follow_mixed_precision():
    r = _only_rule(
        TrainingConfig(mixed_precision="bf16", quantization={"enabled": True})
    )
    # every slot defaults to the compute dtype
    assert r.dtype == {
        "weight": {"fwd": "bf16", "dgrad": "bf16"},
        "act": {"fwd": "bf16", "wgrad": "bf16"},
        "grad_out": {"dgrad": "bf16", "wgrad": "bf16"},
    }
    r32 = _only_rule(
        TrainingConfig(mixed_precision="no", quantization={"enabled": True})
    )
    assert r32.dtype["weight"]["fwd"] == "fp32"


def test_quant_dtype_scalar_applies_to_every_consuming_gemm():
    r = _only_rule(
        TrainingConfig(
            mixed_precision="bf16",
            quantization={"enabled": True, "dtype": {"weight": "fp8_e4m3"}},
        )
    )
    assert r.dtype == {
        "weight": {"fwd": "fp8_e4m3", "dgrad": "fp8_e4m3"},
        "act": {"fwd": "bf16", "wgrad": "bf16"},
        "grad_out": {"dgrad": "bf16", "wgrad": "bf16"},
    }


def test_quant_dtype_scopes_a_tensor_per_gemm():
    # the w16a16dx8 cell: grad_out quantized in dgrad only
    q = QuantizationConfig(
        enabled=True, dtype={"grad_out": {"dgrad": "fp8_e4m3", "wgrad": "bf16"}}
    )
    assert q.dtype["grad_out"] == {"dgrad": "fp8_e4m3", "wgrad": "bf16"}


def test_quant_scoped_dtype_leaves_unset_gemm_to_mixed_precision():
    r = _only_rule(
        TrainingConfig(
            mixed_precision="bf16",
            quantization={"enabled": True, "dtype": {"grad_out": {"dgrad": "int8"}}},
        )
    )
    assert r.dtype["grad_out"] == {"dgrad": "int8", "wgrad": "bf16"}


def test_quant_rejects_a_gemm_that_does_not_consume_the_tensor():
    with pytest.raises(ValueError, match="wgrad"):
        QuantizationConfig(enabled=True, dtype={"weight": {"wgrad": "fp8_e4m3"}})


def test_quant_disabled_rule_dtype_stays_empty():
    # a disabled rule is inert: no operand-dtype fill
    r = _only_rule(
        TrainingConfig(mixed_precision="bf16", quantization={"enabled": False})
    )
    assert r.dtype == {}


def test_quant_dtype_is_mixable():
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "bf16"},
    )
    assert q.dtype["weight"]["fwd"] == "fp8_e4m3"
    assert q.dtype["grad_out"]["wgrad"] == "bf16"


def test_quant_dtype_recipe_fp8():
    q = QuantizationConfig(enabled=True, dtype={"recipe": "fp8"})
    assert q.dtype == {
        "weight": {"fwd": "fp8_e4m3", "dgrad": "fp8_e4m3"},
        "act": {"fwd": "fp8_e4m3", "wgrad": "fp8_e4m3"},
        "grad_out": {"dgrad": "fp8_e5m2", "wgrad": "fp8_e5m2"},
    }


def test_quant_dtype_recipe_respects_an_explicit_tensor():
    q = QuantizationConfig(enabled=True, dtype={"recipe": "fp8", "grad_out": "bf16"})
    assert q.dtype["weight"]["fwd"] == "fp8_e4m3"
    assert q.dtype["grad_out"] == {"dgrad": "bf16", "wgrad": "bf16"}


def test_quant_dtype_recipe_fills_the_gemms_a_scope_leaves_unset():
    # the *_with_gw_hp cell: wgrad opts out, dgrad still follows the recipe
    q = QuantizationConfig(
        enabled=True, dtype={"recipe": "fp8", "grad_out": {"wgrad": "bf16"}}
    )
    assert q.dtype["grad_out"] == {"dgrad": "fp8_e5m2", "wgrad": "bf16"}


def test_quant_unknown_recipe_raises():
    with pytest.raises(ValueError, match="dtype recipe"):
        QuantizationConfig(enabled=True, dtype={"recipe": "fp3"})


def test_quant_include_defaults_empty():
    q = QuantizationConfig()
    assert q.include == [] and q.exclude == ["lm_head", "*mlp.router.gate"]
    q2 = QuantizationConfig(enabled=True, dtype={"recipe": "fp8"}, include=["*.mlp.*"])
    assert q2.include == ["*.mlp.*"]


def test_quant_disabled_skips_validation():
    # inert when off: bad fmt is not checked
    QuantizationConfig(enabled=False, dtype={"weight": "not_a_fmt"})


def test_quant_rejects_unknown_format():
    with pytest.raises(ValueError, match="unknown quant fmt"):
        QuantizationConfig(enabled=True, dtype={"weight": "not_a_fmt"})


def test_quant_accepts_rowwise_granularity():
    q = QuantizationConfig(
        enabled=True, dtype={"recipe": "fp8"}, scale={"granularity": "rowwise"}
    )
    assert q.scale["granularity"] == "rowwise"


def test_quant_default_granularity_is_tensorwise():
    q = QuantizationConfig(enabled=True, dtype={"recipe": "fp8"})
    assert q.scale["granularity"] == "tensorwise"


@pytest.mark.parametrize("granularity", ["tensorwise", "rowwise"])
def test_quant_block_shape_normalized_to_sentinel_off_blockwise(granularity):
    # block_shape only means anything blockwise, so a stale one is zeroed here rather
    # than re-checked against the granularity at every consumer.
    q = QuantizationConfig(
        enabled=True,
        dtype={"recipe": "fp8"},
        scale={"granularity": granularity, "block_shape": (1, 128)},
    )
    assert q.scale["block_shape"] == (0, 0)


def test_quant_recipe_sets_both_backward_grad_slots():
    r = TrainingConfig(
        quantization={"enabled": True, "dtype": {"recipe": "fp8"}}
    ).quantization[0]
    assert r.dtype["grad_out"]["dgrad"] == r.dtype["grad_out"]["wgrad"] == "fp8_e5m2"


def test_quant_wgrad_hp_override():
    # keep only the weight-gradient GEMM in bf16
    q = QuantizationConfig(
        enabled=True, dtype={"recipe": "fp8", "grad_out": {"wgrad": "bf16"}}
    )
    assert q.dtype["grad_out"] == {"dgrad": "fp8_e5m2", "wgrad": "bf16"}


def test_quant_dgrad_hp_override():
    # keep only the input-gradient GEMM in bf16
    q = QuantizationConfig(
        enabled=True, dtype={"recipe": "fp8", "grad_out": {"dgrad": "bf16"}}
    )
    assert q.dtype["grad_out"] == {"dgrad": "bf16", "wgrad": "fp8_e5m2"}


def test_quant_rejects_unknown_dtype_key():
    with pytest.raises(ValueError, match="dtype key"):
        QuantizationConfig(enabled=True, dtype={"bogus_grad": "bf16"})


@pytest.mark.parametrize("retired", ["grad_input", "grad_weight", "dx", "dw"])
def test_quant_rejects_retired_dtype_keys(retired):
    # the old per-GEMM grad keys: the message has to name their replacement
    with pytest.raises(ValueError, match="grad_out"):
        QuantizationConfig(enabled=True, dtype={retired: "bf16"})


def test_quant_rejects_unsupported_granularity():
    with pytest.raises(ValueError, match="granularity"):
        QuantizationConfig(
            enabled=True, dtype={"recipe": "fp8"}, scale={"granularity": "row"}
        )


def test_training_config_normalizes_single_rule_to_list():
    tc = TrainingConfig(quantization={"enabled": True, "dtype": {"recipe": "fp8"}})
    assert isinstance(tc.quantization, list) and len(tc.quantization) == 1
    assert tc.quantization[0].dtype["weight"]["fwd"] == "fp8_e4m3"


def test_training_config_accepts_list_of_rules():
    tc = TrainingConfig(
        mixed_precision="bf16",
        quantization=[
            {"enabled": True, "dtype": {"recipe": "fp8"}, "include": ["*.mlp.*"]},
            {"enabled": True, "dtype": {"weight": "fp8_e4m3"}, "include": ["*.attn.*"]},
        ],
    )
    assert len(tc.quantization) == 2
    assert tc.quantization[0].include == ["*.mlp.*"]
    assert tc.quantization[1].dtype == {
        "weight": {"fwd": "fp8_e4m3", "dgrad": "fp8_e4m3"},
        "act": {"fwd": "bf16", "wgrad": "bf16"},
        "grad_out": {"dgrad": "bf16", "wgrad": "bf16"},
    }


def test_quant_disabled_stays_disabled():
    tc = TrainingConfig(quantization={"enabled": False})
    assert _only_rule(tc).enabled is False


# ---- mxfp8 / blockwise scaling (Option B: element format ⟂ scale scheme) ----


def test_quant_mxfp8_recipe_expands_dtype_and_scale():
    q = QuantizationConfig(enabled=True, dtype={"recipe": "mxfp8"})
    # mxfp8 is e4m3 on every slot (incl. grads): the MX GEMM only does
    # e4m3 x e4m3, and per-block rescaling gives e4m3 enough range for grads.
    assert q.dtype == {
        "weight": {"fwd": "fp8_e4m3", "dgrad": "fp8_e4m3"},
        "act": {"fwd": "fp8_e4m3", "wgrad": "fp8_e4m3"},
        "grad_out": {"dgrad": "fp8_e4m3", "wgrad": "fp8_e4m3"},
    }
    assert q.scale == {
        "granularity": "blockwise",
        "block_shape": (1, 32),
        "scale_dtype": torch.float8_e8m0fnu,
    }


def test_quant_mxfp8_scale_recipe_expands():
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3"},
        scale={"recipe": "mxfp8"},
    )
    assert q.scale["granularity"] == "blockwise"
    assert q.scale["block_shape"] == (1, 32)
    assert q.scale["scale_dtype"] is torch.float8_e8m0fnu
    assert "recipe" not in q.scale  # recipe key is consumed on expansion


def test_quant_mxfp8_scale_recipe_explicit_keys_win():
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3"},
        scale={"recipe": "mxfp8", "block_shape": (1, 64)},
    )
    assert q.scale["block_shape"] == (1, 64)  # explicit overrides recipe default


def test_quant_unknown_scale_recipe_raises():
    with pytest.raises(ValueError, match="scale recipe"):
        QuantizationConfig(
            enabled=True, dtype={"weight": "fp8_e4m3"}, scale={"recipe": "nope"}
        )


def test_quant_e8m0_requires_blockwise():
    with pytest.raises(ValueError, match="fp8_e8m0"):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            scale={"granularity": "rowwise", "scale_dtype": "fp8_e8m0"},
        )


def test_quant_blockwise_requires_block_size():
    with pytest.raises(ValueError, match="block_shape"):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            scale={"granularity": "blockwise"},
        )


def test_quant_mxfp8_rejects_int_element():
    with pytest.raises(ValueError, match="fp8"):
        QuantizationConfig(
            enabled=True, dtype={"weight": "int8"}, scale={"recipe": "mxfp8"}
        )


@pytest.mark.parametrize("scale_dtype", ({}, {"scale_dtype": None}))
def test_quant_rowwise_defaults_scale_dtype_to_fp32(scale_dtype):
    # adding blockwise validation must not disturb the existing rowwise path
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3"},
        scale={"granularity": "rowwise", **scale_dtype},
    )
    assert q.scale["granularity"] == "rowwise"
    assert q.scale["scale_dtype"] is torch.float32


def test_quant_mxfp8_normalizes_through_training_config():
    r = _only_rule(
        TrainingConfig(
            mixed_precision="bf16",
            quantization={"dtype": {"recipe": "mxfp8"}, "enabled": True},
        )
    )
    assert r.scale["scale_dtype"] is torch.float8_e8m0fnu
    assert r.dtype["weight"]["fwd"] == "fp8_e4m3"


# ==================== latent_moe / latent_dim ====================


def test_modelconfig_latent_moe_default_off():
    cfg = ModelConfig(
        d_model=64,
        mlp=[
            {"mlp_cls": "moe", "mlp_kwargs": {"n_routed_experts": 4, "aux_loss": True}}
        ],
    )
    assert cfg.resolve_mlp(0)[1]["latent_moe"] is False
    assert "latent_dim" not in cfg.resolve_mlp(0)[1]


@pytest.mark.parametrize("latent_dim", [None, 0, -4, 3.5])
def test_modelconfig_latent_moe_requires_positive_latent_dim(latent_dim):
    kwargs = {"n_routed_experts": 4, "aux_loss": True, "latent_moe": True}
    if latent_dim is not None:
        kwargs["latent_dim"] = latent_dim
    with pytest.raises(ValueError, match="latent_dim must be a positive int"):
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


def _moe_kwargs(**over):
    base = {"n_routed_experts": 4, "expert_bias": True}
    base.update(over)
    return base


def test_mlp_single_item_covers_all_layers():
    cfg = ModelConfig(
        d_model=64, n_layers=4, mlp=[{"mlp_cls": "moe", "mlp_kwargs": _moe_kwargs()}]
    )
    assert [cfg.resolve_mlp(i)[0] for i in range(cfg.n_layers)] == ["moe"] * 4
    assert cfg.is_moe is True
    # per-item defaulting ran
    assert cfg.resolve_mlp(0)[1]["intermediate_size"] == 4 * 64


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
    with pytest.raises(ValueError, match="claimed by multiple"):
        ModelConfig(
            d_model=64,
            n_layers=4,
            mlp=[
                {"mlp_cls": "dense", "layer_idx": [0]},
                {"mlp_cls": "moe", "mlp_kwargs": _moe_kwargs(), "layer_idx": [0, 1]},
                {"mlp_cls": "dense"},
            ],
        )


def test_mlp_within_item_duplicate_raises_clear_message():
    with pytest.raises(ValueError, match="listed more than once in one mlp item"):
        ModelConfig(
            d_model=64,
            n_layers=4,
            mlp=[
                {"mlp_cls": "dense", "layer_idx": [0, 0]},
                {"mlp_cls": "dense"},
            ],
        )


def test_mlp_gap_raises():
    with pytest.raises(ValueError, match="no mlp item|not claimed|coverage"):
        ModelConfig(
            d_model=64,
            n_layers=4,
            mlp=[
                {"mlp_cls": "dense", "layer_idx": [0]},
                {"mlp_cls": "moe", "mlp_kwargs": _moe_kwargs(), "layer_idx": [1]},
            ],
        )


def test_mlp_two_bare_items_raise():
    with pytest.raises(ValueError, match="at most one"):
        ModelConfig(
            d_model=64,
            n_layers=4,
            mlp=[
                {"mlp_cls": "dense"},
                {"mlp_cls": "dense"},
            ],
        )


def test_mlp_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
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
    assert (
        not hasattr(cfg, "aux_loss_coef")
        or callable(getattr(type(cfg), "aux_loss_coef", None)) is False
    )


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


# ==================== Scaling recipes (tensorwise, rowwise, blockwise) ====================


def test_quantization_config_rotation_defaults():
    """Config canonicalization applies defaults without storing runtime state."""
    q = QuantizationConfig(
        enabled=True,
        dtype={"recipe": "fp8"},
        rotation={"rotation_cls": "hadamard"},
    )
    assert q.rotation["rotation_cls"] == "hadamard"
    assert q.rotation["gemms"] == ["fwd", "dgrad", "wgrad"]
    assert q.rotation["rotation_kwargs"] == {}
    exported = TrainConfig(training=TrainingConfig(quantization=q)).to_dict()
    assert exported["training"]["quantization"][0]["rotation"] == q.rotation
    yaml.safe_dump(exported)


@pytest.mark.parametrize(
    "rotation",
    [
        None,
        {},
        {"rotation_kwargs": {"block_size": 32}},
        {"gemms": ["wgrad"]},
    ],
)
def test_quantization_config_rotation_requires_rotation_cls(rotation):
    q = QuantizationConfig(enabled=True, dtype={"recipe": "fp8"}, rotation=rotation)
    assert q.rotation is None


@pytest.mark.parametrize("block_size", [1, 2, 32, 128])
def test_quantization_config_rotation(block_size):
    q = QuantizationConfig(
        enabled=True,
        dtype={"recipe": "fp8"},
        rotation={
            "rotation_cls": "hadamard",
            "rotation_kwargs": {"block_size": block_size},
        },
    )
    assert q.rotation["rotation_cls"] == "hadamard"
    assert q.rotation["rotation_kwargs"]["block_size"] == block_size


@pytest.mark.parametrize(
    "gemms, expected",
    [
        (["wgrad"], ["wgrad"]),
        (["dgrad", "wgrad"], ["dgrad", "wgrad"]),
        ("wgrad", ["wgrad"]),
    ],
)
def test_quantization_config_rotation_gemms(gemms, expected):
    q = QuantizationConfig(
        enabled=True,
        dtype={"recipe": "fp8"},
        rotation={"rotation_cls": "hadamard", "gemms": gemms},
    )
    assert q.rotation["gemms"] == expected


@pytest.mark.parametrize(
    "rotation",
    [
        [],
        {"rotation_cls": []},
        {"rotation_cls": "givens"},
        {"rotation_cls": "hadamard", "rotation_kwargs": 4},
        {"rotation_cls": "hadamard", "rotation_kwargs": []},
        {"rotation_cls": "hadamard", "gemms": ["bwd"]},
        {"rotation_cls": "hadamard", "gemms": ["wgrad", "wgrad"]},
        {"rotation_cls": "hadamard", "gemm": ["wgrad"]},
    ],
)
def test_quantization_config_rotation_raise_error(rotation):
    with pytest.raises(ValueError, match="rotation"):
        QuantizationConfig(
            enabled=True,
            dtype={"recipe": "fp8"},
            rotation=rotation,
        )


def test_training_config_keeps_per_rule_rotations():
    """Rules must be free to rotate differently; each keeps its own spec."""
    config = TrainingConfig(
        mixed_precision="bf16",
        quantization=[
            {
                "enabled": True,
                "dtype": {"recipe": "int8"},
                "include": ["*attn*"],
                "exclude": [],
                "rotation": {
                    "rotation_cls": "hadamard",
                    "rotation_kwargs": {"block_size": 16, "seed": 1},
                },
            },
            {
                "enabled": True,
                "dtype": {"recipe": "int8"},
                "include": ["*mlp*"],
                "exclude": [],
                "rotation": {
                    "rotation_cls": "hadamard",
                    "rotation_kwargs": {"block_size": 64, "seed": 2},
                    "gemms": ["wgrad"],
                },
            },
        ],
    )

    attn, mlp = config.quantization
    assert attn.rotation["rotation_kwargs"] == {"block_size": 16, "seed": 1}
    assert attn.rotation["gemms"] == ["fwd", "dgrad", "wgrad"]
    assert mlp.rotation["rotation_kwargs"] == {"block_size": 64, "seed": 2}
    assert mlp.rotation["gemms"] == ["wgrad"]


def test_quantization_config_rotation_kwargs_raise_error():
    """Kwargs must survive the YAML round trip that carries them into a run."""
    with pytest.raises(ValueError, match="rotation"):
        QuantizationConfig(
            enabled=True,
            dtype={"recipe": "fp8"},
            rotation={
                "rotation_cls": "hadamard",
                "rotation_kwargs": {"sign_vector": torch.ones(4)},
            },
        )


@pytest.mark.parametrize(
    "rotation_kwargs, expected_seed",
    [
        ({}, 23),
        ({"seed": 7}, 7),
    ],
)
def test_training_config_hadamard_rotation_seed(rotation_kwargs, expected_seed):
    """An omitted Hadamard seed must inherit the training seed."""
    config = TrainingConfig(
        mixed_precision="bf16",
        seed=23,
        quantization={
            "enabled": True,
            "dtype": {"recipe": "int8"},
            "rotation": {
                "rotation_cls": "hadamard",
                "rotation_kwargs": rotation_kwargs,
            },
        },
    )

    assert _only_rule(config).rotation["rotation_kwargs"]["seed"] == expected_seed


def test_quant_blockwise_recipe_expands():
    q = QuantizationConfig(
        enabled=True, dtype={"recipe": "fp8"}, scale={"recipe": "blockwise"}
    )
    assert q.scale["granularity"] == "blockwise"
    assert q.scale["block_shape"] == (1, 128)
    assert q.scale["scale_dtype"] is torch.float32
    assert "recipe" not in q.scale


def test_quant_rowwise_recipe_expands():
    q = QuantizationConfig(
        enabled=True, dtype={"recipe": "fp8"}, scale={"recipe": "rowwise"}
    )
    assert q.scale["granularity"] == "rowwise"


def test_quant_blockwise_block_size_must_be_multiple_of_16():
    with pytest.raises(ValueError, match="block_shape"):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            scale={"granularity": "blockwise", "block_shape": (1, 24)},
        )


def test_quant_blockwise_block_size_must_be_positive():
    with pytest.raises(ValueError, match="block_shape"):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            scale={"granularity": "blockwise", "block_shape": (1, -16)},
        )


def test_quant_rejects_unknown_scale_dtype():
    with pytest.raises(ValueError, match="scale_dtype"):
        QuantizationConfig(
            enabled=True,
            dtype={"weight": "fp8_e4m3"},
            scale={
                "granularity": "blockwise",
                "block_shape": (1, 32),
                "scale_dtype": "e3m4",
            },
        )


def test_quant_blockwise_fp32_scale_dtype_ok():
    q = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3"},
        scale={
            "granularity": "blockwise",
            "block_shape": (1, 64),
            "scale_dtype": "fp32",
        },
    )
    assert q.scale["scale_dtype"] is torch.float32


def test_monitoring_flags_default_true():
    from src.utils.config import LoggingConfig

    assert LoggingConfig().log_quant_metrics is True
    assert LoggingConfig().log_activation_norms is True


@pytest.mark.parametrize("tile", [16, 32, 64])
def test_quant_block_shape_accepts_square(tile):
    q = QuantizationConfig(
        enabled=True,
        dtype={"recipe": "fp8"},
        scale={
            "granularity": "blockwise",
            "block_shape": (tile, tile),
            "scale_dtype": "fp32",
        },
    )
    assert q.scale["block_shape"] == (tile, tile)


def test_quant_block_shape_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        QuantizationConfig(
            enabled=True,
            dtype={"recipe": "fp8"},
            scale={
                "granularity": "blockwise",
                "block_shape": (16, 128),
                "scale_dtype": "fp32",
            },
        )


def test_quant_e8m0_requires_contract_extent_multiple_of_32():
    with pytest.raises(ValueError, match="32"):
        QuantizationConfig(
            enabled=True,
            dtype={"recipe": "mxfp8"},
            scale={
                "granularity": "blockwise",
                "block_shape": (16, 16),
                "scale_dtype": "fp8_e8m0",
            },
        )
