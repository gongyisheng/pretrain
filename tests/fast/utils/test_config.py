import tempfile
import os
import pytest
import yaml
from src.utils.config import ModelConfig, load_config, TrainingConfig, DataConfig


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
    extra = {"n_experts": 4} if mlp_cls == "moe" else {}
    cfg = ModelConfig(d_model=128, mlp_cls=mlp_cls, mlp_kwargs=dict(extra))
    assert cfg.mlp_kwargs["intermediate_size"] == 4 * 128
    # explicit value preserved
    cfg2 = ModelConfig(
        d_model=128, mlp_cls=mlp_cls, mlp_kwargs={"intermediate_size": 256, **extra}
    )
    assert cfg2.mlp_kwargs["intermediate_size"] == 256


def test_modelconfig_moe_expert_bias_defaults():
    # default: aux-loss balancing, no expert_bias
    cfg = ModelConfig(d_model=64, mlp_cls="moe", mlp_kwargs={"n_experts": 4})
    assert cfg.mlp_kwargs["expert_bias"] is False
    assert cfg.mlp_kwargs["aux_loss"] is True
    assert cfg.mlp_kwargs["aux_loss_coef"] == 0.01
    # expert_bias on: aux_loss defaults off, bias update rate defaulted
    cfg2 = ModelConfig(
        d_model=64, mlp_cls="moe", mlp_kwargs={"n_experts": 4, "expert_bias": True}
    )
    assert cfg2.mlp_kwargs["aux_loss"] is False
    assert cfg2.mlp_kwargs["expert_bias_update_rate"] == 0.001
    assert "aux_loss_coef" not in cfg2.mlp_kwargs


def test_modelconfig_moe_aux_loss_and_expert_bias_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelConfig(
            d_model=64,
            mlp_cls="moe",
            mlp_kwargs={"n_experts": 4, "aux_loss": True, "expert_bias": True},
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


def test_training_unknown_fp8_recipe_raises():
    with pytest.raises(ValueError, match="unknown fp8_recipe"):
        TrainingConfig(fp8=True, fp8_recipe="not_a_real_recipe")


def test_training_fp8_recipe_unchecked_when_disabled():
    # Recipe is only validated when fp8 is enabled.
    TrainingConfig(fp8=False, fp8_recipe="not_a_real_recipe")
