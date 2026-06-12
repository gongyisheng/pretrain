import tempfile
import os
import yaml
from src.utils.config import ModelConfig, load_config, ModelTrainingConfig, DataConfig


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
    assert cfg.attn_kwargs == {}
    assert cfg.mlp_kwargs == {}
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


# ==================== ModelTrainingConfig / DataConfig ====================


def test_training_config_intra_doc_masking_default():
    cfg = ModelTrainingConfig()
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
    assert tc.method_kwargs == {}
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
    from src.utils.config import ModelTrainingConfig

    cfg = ModelTrainingConfig()
    assert cfg.eval_train is False


# ==================== attn_cls / attn_kwargs ====================


def test_attn_cls_defaults():
    # Default attn_cls is gqa
    assert ModelConfig().attn_cls == "gqa"
    assert ModelConfig(attn_cls="mha").attn_cls == "mha"
    assert ModelConfig(attn_cls="mla").attn_cls == "mla"


def test_attn_kwargs_preserved():
    # attn_kwargs defaults to empty; explicit values are preserved
    assert ModelConfig().attn_kwargs == {}
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
