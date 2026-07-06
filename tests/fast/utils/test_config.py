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
        "attn_cls": "gqa",
        "attn_kwargs": {
            "n_heads": 4,
            "dropout": 0.0,
            "attn_implementation": "flex_attention",
        },
        "mlp_cls": "dense",
        "mlp_kwargs": {"activation": "silu", "gated": True, "intermediate_size": 0},
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
    assert cfg.attn_cls == "gqa"
    assert cfg.mlp_cls == "dense"
    assert cfg.norm_cls == "rmsnorm"
    assert cfg.pos_emb_cls == "rope"
    assert cfg.residual_cls == "standard"
    # __post_init__ fills component defaults: attn_implementation for attn,
    # intermediate_size (4*d_model) for mlp.
    assert cfg.attn_kwargs == {"attn_implementation": "flex_attention"}
    assert cfg.mlp_kwargs == {"intermediate_size": 4 * 768}
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
        assert cfg.model.attn_kwargs["n_heads"] == 4
        assert cfg.model.mlp_kwargs["activation"] == "silu"
        assert cfg.model.pos_emb_kwargs["rope_theta"] == 1e4


def test_config_to_dict_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        d = config.to_dict()
        assert d["max_seq_len"] == 128
        assert d["model"]["attn_cls"] == "gqa"
        assert d["model"]["attn_kwargs"]["n_heads"] == 4


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
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        cfg = load_config(path, overrides=["model.attn_kwargs.n_heads=8"])
        assert cfg.model.attn_kwargs["n_heads"] == 8


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
    assert ModelConfig().attn_cls == "gqa"
    assert ModelConfig(attn_cls="mha").attn_cls == "mha"
    assert ModelConfig(attn_cls="mla").attn_cls == "mla"


def test_attn_kwargs_preserved():
    # attn_implementation default is filled; explicit values are preserved
    assert ModelConfig().attn_kwargs == {"attn_implementation": "flex_attention"}
    assert ModelConfig(attn_kwargs={"n_heads": 8}).attn_kwargs["n_heads"] == 8


def test_attn_kwargs_round_trip_from_yaml(tmp_path):
    yaml_content = """
model:
  arch: qwen3
  attn_cls: mla
  d_model: 64
  attn_kwargs:
    n_heads: 8
    kv_lora_rank: 32
    qk_rope_head_dim: 16
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.model.attn_cls == "mla"
    assert cfg.model.attn_kwargs["kv_lora_rank"] == 32
    assert cfg.model.attn_kwargs["qk_rope_head_dim"] == 16
    assert cfg.model.attn_kwargs["n_heads"] == 8


# ==================== component defaults + validation (moved into config) ====================


@pytest.mark.parametrize("mlp_cls", ["dense", "moe"])
def test_modelconfig_resolves_intermediate_size(mlp_cls):
    extra = {"n_routed_experts": 4} if mlp_cls == "moe" else {}
    cfg = ModelConfig(d_model=128, mlp_cls=mlp_cls, mlp_kwargs=dict(extra))
    assert cfg.mlp_kwargs["intermediate_size"] == 4 * 128
    # explicit value preserved
    cfg2 = ModelConfig(
        d_model=128, mlp_cls=mlp_cls, mlp_kwargs={"intermediate_size": 256, **extra}
    )
    assert cfg2.mlp_kwargs["intermediate_size"] == 256


def test_modelconfig_moe_expert_bias_defaults():
    # default: aux-loss balancing, no expert_bias
    cfg = ModelConfig(d_model=64, mlp_cls="moe", mlp_kwargs={"n_routed_experts": 4})
    assert cfg.mlp_kwargs["expert_bias"] is False
    assert cfg.mlp_kwargs["aux_loss"] is True
    assert cfg.mlp_kwargs["aux_loss_coef"] == 0.01
    # expert_bias on: aux_loss defaults off, bias update rate defaulted
    cfg2 = ModelConfig(
        d_model=64,
        mlp_cls="moe",
        mlp_kwargs={"n_routed_experts": 4, "expert_bias": True},
    )
    assert cfg2.mlp_kwargs["aux_loss"] is False
    assert cfg2.mlp_kwargs["expert_bias_update_rate"] == 0.001
    assert "aux_loss_coef" not in cfg2.mlp_kwargs


def test_modelconfig_moe_aux_loss_and_expert_bias_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelConfig(
            d_model=64,
            mlp_cls="moe",
            mlp_kwargs={"n_routed_experts": 4, "aux_loss": True, "expert_bias": True},
        )


def test_modelconfig_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        ModelConfig(mlp_kwargs={"activation": "mish"})


def test_modelconfig_gated_only_activation_rejected_when_ungated():
    # bilinear is gated-only; rejected for an ungated mlp
    with pytest.raises(ValueError, match="Unknown activation"):
        ModelConfig(mlp_kwargs={"activation": "bilinear", "gated": False})
    # accepted when gated
    ModelConfig(mlp_kwargs={"activation": "bilinear", "gated": True})


def test_modelconfig_validates_attn_dims():
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ModelConfig(d_model=100, attn_kwargs={"n_heads": 3})
    with pytest.raises(ValueError, match="divisible by\\s+n_kv_heads"):
        ModelConfig(
            d_model=64, attn_cls="gqa", attn_kwargs={"n_heads": 4, "n_kv_heads": 3}
        )


def test_modelconfig_gqa_defaults_n_kv_heads():
    cfg = ModelConfig(d_model=64, attn_cls="gqa", attn_kwargs={"n_heads": 8})
    assert cfg.attn_kwargs["n_kv_heads"] == 8  # defaults to n_heads


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


def _moe_train_config(mixed_precision, expert_capacity_factor=None):
    """Build a minimal TrainConfig with dropless (or capped) MoE."""
    mlp_kwargs = {
        "n_routed_experts": 4,
        "n_routed_experts_per_token": 2,
        "intermediate_size": 64,
    }
    if expert_capacity_factor is not None:
        mlp_kwargs["expert_capacity_factor"] = expert_capacity_factor
    return TrainConfig(
        model=ModelConfig(d_model=64, mlp_cls="moe", mlp_kwargs=mlp_kwargs),
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


def test_capacity_capped_moe_any_precision_ok():
    # expert_capacity_factor set → capacity-capped path; no restriction
    _moe_train_config("fp16", expert_capacity_factor=1.25)
    _moe_train_config("no", expert_capacity_factor=1.25)
    _moe_train_config("bf16", expert_capacity_factor=1.25)


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
