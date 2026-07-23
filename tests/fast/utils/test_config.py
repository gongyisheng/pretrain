import glob
import tempfile
import os
import pytest
import yaml
from src.utils.config import (
    ModelConfig,
    TrainConfig,
    load_config,
    TrainingConfig,
    DataConfig,
    QuantConfig,
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
                    "activation": "silu",
                    "gated": True,
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
        "activation_checkpointing": False,
        "grad_clip": 1.0,
        "checkpoint_dir": "ckpt/",
        "checkpoint_every": 5,
        "eval_every": 5,
        "eval_steps": 2,
    },
    "optimizer": {
        "name": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
    },
    "scheduler": {"name": "cosine", "warmup_steps": 2, "min_lr": 1e-4},
    "logging": {"wandb_project": "test", "wandb_run_name": "test", "log_every": 1},
}


# ==================== ModelConfig defaults ====================


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.layer_attn_classes()[0] == "gqa"
    assert cfg.layer_mlp_classes()[0] == "dense"
    assert cfg.norm_cls == "rmsnorm"
    assert cfg.pos_emb_cls == "rope"
    assert cfg.residual_cls == "standard"
    # __post_init__ fills component defaults: attn_implementation for attn,
    # intermediate_size (4*d_model) for mlp.
    assert cfg.resolve_attn(0)[1] == {"attn_implementation": "flex_attention"}
    assert cfg.resolve_mlp(0)[1] == {"intermediate_size": 4 * 768}
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
        assert cfg.model.resolve_mlp(0)[1]["activation"] == "silu"
        assert cfg.model.pos_emb_kwargs["rope_theta"] == 1e4


def test_config_to_dict_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        d = config.to_dict()
        assert d["max_seq_len"] == 128
        assert d["model"]["attn"][0]["attn_cls"] == "gqa"
        assert d["model"]["attn"][0]["attn_kwargs"]["n_heads"] == 4


# ==================== CLI overrides ====================


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


# ==================== attn_cls / attn_kwargs ====================


def test_attn_cls_defaults():
    # Default attn_cls is gqa
    assert ModelConfig().layer_attn_classes()[0] == "gqa"
    assert (
        ModelConfig(attn=[{"attn_cls": "mha", "attn_kwargs": {}}]).layer_attn_classes()[
            0
        ]
        == "mha"
    )
    assert (
        ModelConfig(attn=[{"attn_cls": "mla", "attn_kwargs": {}}]).layer_attn_classes()[
            0
        ]
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
    assert cfg.model.layer_attn_classes()[0] == "mla"
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
    assert cfg.layer_attn_classes() == ["gqa"] * 4
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
    assert cfg.layer_attn_classes() == ["mha", "gqa", "gqa", "gqa"]
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
    with pytest.raises(ValueError, match="router_score_fn must be one of"):
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


def test_modelconfig_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        ModelConfig(mlp=[{"mlp_cls": "dense", "mlp_kwargs": {"activation": "mish"}}])


def test_modelconfig_gated_only_activation_rejected_when_ungated():
    # bilinear is gated-only; rejected for an ungated mlp
    with pytest.raises(ValueError, match="Unknown activation"):
        ModelConfig(
            mlp=[
                {
                    "mlp_cls": "dense",
                    "mlp_kwargs": {"activation": "bilinear", "gated": False},
                }
            ]
        )
    # accepted when gated
    ModelConfig(
        mlp=[
            {
                "mlp_cls": "dense",
                "mlp_kwargs": {"activation": "bilinear", "gated": True},
            }
        ]
    )


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


def test_optimizer_unknown_name_raises():
    from src.utils.config import OptimizerConfig

    with pytest.raises(ValueError, match="unknown optimizer"):
        OptimizerConfig(name="sgd")
    OptimizerConfig(name="adamw")
    OptimizerConfig(name="lion")


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


def test_dropless_moe_requires_bf16_fp16_raises():
    with pytest.raises(ValueError, match="bf16"):
        _moe_train_config("fp16")


def test_dropless_moe_requires_bf16_no_raises():
    with pytest.raises(ValueError, match="bf16"):
        _moe_train_config("no")


def test_dropless_moe_bf16_ok():
    cfg = _moe_train_config("bf16")
    assert cfg.training.mixed_precision == "bf16"


# ==================== Quantization config ====================


def _only_rule(tc):
    # TrainingConfig normalizes quant to a list of rules; single-rule helper.
    assert len(tc.quant) == 1
    return tc.quant[0]


def test_quant_defaults_disabled():
    q = QuantConfig()
    assert q.enabled is False
    assert q.exclude == ["lm_head"]
    # disabled rule is inert: no dtype/scaling defaults applied
    assert q.dtype == {} and q.scaling == {}


def test_quant_operand_defaults_follow_mixed_precision():
    r = _only_rule(TrainingConfig(mixed_precision="bf16", quant={"enabled": True}))
    # every operand defaults to the compute dtype
    assert r.dtype == {
        "weight": "bf16",
        "act": "bf16",
        "input_grad": "bf16",
        "weight_grad": "bf16",
    }
    r32 = _only_rule(TrainingConfig(mixed_precision="no", quant={"enabled": True}))
    assert r32.dtype["weight"] == "fp32"
    # explicit operands are kept; only unset ones follow the compute dtype
    r2 = _only_rule(
        TrainingConfig(
            mixed_precision="bf16",
            quant={"enabled": True, "dtype": {"weight": "fp8"}},
        )
    )
    assert r2.dtype == {
        "weight": "fp8",
        "act": "bf16",
        "input_grad": "bf16",
        "weight_grad": "bf16",
    }


def test_quant_disabled_rule_dtype_stays_empty():
    # a disabled rule is inert: no operand-dtype fill
    r = _only_rule(TrainingConfig(mixed_precision="bf16", quant={"enabled": False}))
    assert r.dtype == {}


def test_quant_dtype_is_mixable():
    q = QuantConfig(
        enabled=True, dtype={"weight": "fp8", "act": "fp8", "weight_grad": "bf16"}
    )
    assert q.dtype["weight"] == "fp8" and q.dtype["weight_grad"] == "bf16"


def test_quant_dtype_recipe_fp8():
    q = QuantConfig(enabled=True, dtype_recipe="fp8")
    assert q.dtype == {
        "weight": "fp8_e4m3",
        "act": "fp8_e4m3",
        "input_grad": "fp8_e5m2",
        "weight_grad": "fp8_e5m2",
    }


def test_quant_dtype_recipe_respects_explicit_operand():
    # explicit weight_grad: bf16 overrides the recipe (torchao *_with_gw_hp)
    q = QuantConfig(enabled=True, dtype_recipe="fp8", dtype={"weight_grad": "bf16"})
    assert q.dtype["weight"] == "fp8_e4m3" and q.dtype["weight_grad"] == "bf16"


def test_quant_unknown_recipe_raises():
    with pytest.raises(ValueError, match="dtype_recipe"):
        QuantConfig(enabled=True, dtype_recipe="fp3")


def test_quant_include_defaults_empty():
    q = QuantConfig()
    assert q.include == [] and q.exclude == ["lm_head"]
    q2 = QuantConfig(enabled=True, dtype_recipe="fp8", include=["*.mlp.*"])
    assert q2.include == ["*.mlp.*"]


def test_quant_disabled_skips_validation():
    # inert when off: bad fmt is not checked
    QuantConfig(enabled=False, dtype={"weight": "not_a_fmt"})


def test_quant_rejects_unknown_format():
    with pytest.raises(ValueError, match="unknown quant fmt"):
        QuantConfig(enabled=True, dtype={"weight": "not_a_fmt"})


def test_quant_accepts_rowwise_granularity():
    q = QuantConfig(
        enabled=True, dtype_recipe="fp8", scaling={"granularity": "rowwise"}
    )
    assert q.scaling["granularity"] == "rowwise"


def test_quant_default_granularity_is_tensorwise():
    q = QuantConfig(enabled=True, dtype_recipe="fp8")
    assert q.scaling["granularity"] == "tensorwise"


def test_quant_recipe_sets_both_backward_grads():
    r = TrainingConfig(quant={"enabled": True, "dtype_recipe": "fp8"}).quant[0]
    assert r.dtype["input_grad"] == r.dtype["weight_grad"] == "fp8_e5m2"
    assert "grad" not in r.dtype


def test_quant_weight_grad_hp_override():
    # gw_hp: keep only the weight-gradient GEMM in bf16
    q = QuantConfig(enabled=True, dtype_recipe="fp8", dtype={"weight_grad": "bf16"})
    assert q.dtype["weight_grad"] == "bf16" and q.dtype["input_grad"] == "fp8_e5m2"


def test_quant_input_grad_hp_override():
    # keep only the input-gradient (dgrad) GEMM in bf16
    q = QuantConfig(enabled=True, dtype_recipe="fp8", dtype={"input_grad": "bf16"})
    assert q.dtype["input_grad"] == "bf16" and q.dtype["weight_grad"] == "fp8_e5m2"


def test_quant_rejects_unknown_dtype_key():
    with pytest.raises(ValueError, match="dtype key"):
        QuantConfig(enabled=True, dtype={"grad_weight": "bf16"})


def test_quant_rejects_unsupported_granularity():
    with pytest.raises(ValueError, match="granularity"):
        QuantConfig(enabled=True, dtype_recipe="fp8", scaling={"granularity": "row"})


def test_training_config_normalizes_single_rule_to_list():
    tc = TrainingConfig(quant={"enabled": True, "dtype_recipe": "fp8"})
    assert isinstance(tc.quant, list) and len(tc.quant) == 1
    assert tc.quant[0].dtype["weight"] == "fp8_e4m3"


def test_training_config_accepts_list_of_rules():
    tc = TrainingConfig(
        mixed_precision="bf16",
        quant=[
            {"enabled": True, "dtype_recipe": "fp8", "include": ["*.mlp.*"]},
            {"enabled": True, "dtype": {"weight": "fp8"}, "include": ["*.attn.*"]},
        ],
    )
    assert len(tc.quant) == 2
    assert tc.quant[0].include == ["*.mlp.*"]
    assert tc.quant[1].dtype == {
        "weight": "fp8",
        "act": "bf16",
        "input_grad": "bf16",
        "weight_grad": "bf16",
    }


def test_quant_disabled_stays_disabled():
    tc = TrainingConfig(quant={"enabled": False})
    assert _only_rule(tc).enabled is False


# ==================== per-layer mlp schema + resolver ====================


def _moe_kwargs(**over):
    base = {"n_routed_experts": 4, "expert_bias": True}
    base.update(over)
    return base


def test_mlp_single_item_covers_all_layers():
    cfg = ModelConfig(
        d_model=64, n_layers=4, mlp=[{"mlp_cls": "moe", "mlp_kwargs": _moe_kwargs()}]
    )
    assert cfg.layer_mlp_classes() == ["moe"] * 4
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
    assert cfg.layer_mlp_classes() == ["dense", "moe", "moe", "moe"]
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


def test_mlp_shared_aux_coef_mismatch_raises():
    with pytest.raises(ValueError, match="aux_loss_coef"):
        ModelConfig(
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


def test_all_configs_load_and_have_list_mlp():
    paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
        "experiments/**/*.yaml", recursive=True
    )
    assert paths
    for p in paths:
        cfg = load_config(p)
        assert isinstance(cfg.model.mlp, list) and cfg.model.mlp
        # resolver ran and covers every layer
        assert len(cfg.model.layer_mlp_classes()) == cfg.model.n_layers
        assert isinstance(cfg.model.attn, list) and cfg.model.attn
        assert len(cfg.model.layer_attn_classes()) == cfg.model.n_layers


def test_configs_model_key_order_attn_mlp_first():
    for p in ("configs/gpt2_124m.yaml", "configs/qwen3_51m.yaml"):
        raw = yaml.safe_load(open(p))
        keys = list(raw["model"].keys())
        assert keys[0] == "attn" and keys[1] == "mlp", (p, keys)
