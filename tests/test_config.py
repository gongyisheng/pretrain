import pytest
import tempfile
import os
import yaml
from src.utils.config import TrainConfig, load_config, TrainingConfig, DataConfig


def _write_yaml(tmp_dir, data):
    path = os.path.join(tmp_dir, "test.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


MINIMAL_CONFIG = {
    "max_seq_len": 128,
    "model": {"arch": "gpt2", "n_layers": 2, "n_heads": 2, "d_model": 64, "vocab_size": 256, "dropout": 0.0},
    "data": {"dataset": "test", "tokenizer_path": "tok", "data_dir": "data/", "val_split": 0.01, "num_workers": 0},
    "training": {
        "batch_size": 2, "gradient_accumulation_steps": 1, "max_steps": 10,
        "mixed_precision": "no", "activation_checkpointing": False,
        "grad_clip": 1.0, "checkpoint_dir": "ckpt/", "checkpoint_every": 5,
        "eval_every": 5, "eval_steps": 2,
    },
    "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.1, "betas": [0.9, 0.95]},
    "scheduler": {"name": "cosine", "warmup_steps": 2, "min_lr": 1e-4},
    "logging": {"wandb_project": "test", "wandb_run_name": "test", "log_every": 1},
}


def test_load_config_from_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        assert config.max_seq_len == 128
        assert config.model.arch == "gpt2"
        assert config.model.n_layers == 2
        assert config.optimizer.lr == 1e-3


def test_config_d_ff_default():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        assert config.model.intermediate_size == 4 * 64


def test_config_d_ff_explicit():
    data = {**MINIMAL_CONFIG, "model": {**MINIMAL_CONFIG["model"], "intermediate_size": 128}}
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, data)
        config = load_config(path)
        assert config.model.intermediate_size == 128


def test_config_cli_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path, overrides=["optimizer.lr=3e-4", "training.batch_size=8"])
        assert config.optimizer.lr == 3e-4
        assert config.training.batch_size == 8


def test_config_to_dict_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        d = config.to_dict()
        assert d["max_seq_len"] == 128
        assert d["model"]["arch"] == "gpt2"


def test_config_moe_fields():
    data = {
        **MINIMAL_CONFIG,
        "model": {
            **MINIMAL_CONFIG["model"],
            "n_experts": 8,
            "n_experts_per_token": 2,
            "moe_aux_loss_coef": 0.02,
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, data)
        config = load_config(path)
        assert config.model.n_experts == 8
        assert config.model.n_experts_per_token == 2
        assert config.model.moe_aux_loss_coef == 0.02


def test_config_moe_fields_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        assert config.model.n_experts == 0
        assert config.model.n_experts_per_token == 2
        assert config.model.moe_aux_loss_coef == 0.01


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
    """Deprecated or unknown YAML fields should be silently ignored."""
    yaml_content = """
max_seq_len: 128
data:
  eot_token_id: 0
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.max_seq_len == 128
