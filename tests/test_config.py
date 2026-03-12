import pytest
import tempfile
import os
import yaml
from src.utils.config import TrainConfig, load_config


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
        assert config.model.d_ff == 4 * 64


def test_config_d_ff_explicit():
    data = {**MINIMAL_CONFIG, "model": {**MINIMAL_CONFIG["model"], "d_ff": 128}}
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, data)
        config = load_config(path)
        assert config.model.d_ff == 128


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
