"""Dry-run training tests for GPT-2 and Qwen3 using real configs and data."""

import os
import sys

import pytest

sys.path.insert(0, ".")

from src.utils.config import load_config
from src.training.trainer import Trainer


STEPS = 1

_TOKENIZER_PATH = "tokenizers/custom_bpe_50k/tokenizer.json"


def _require_tokenizer():
    if not os.path.exists(_TOKENIZER_PATH):
        pytest.fail(
            f"tokenizer not found at {_TOKENIZER_PATH}; run preprocess_data first"
        )


def _run_dry_run(config_path):
    overrides = [
        f"training.early_stop={STEPS}",
        f"training.eval_every={STEPS + 1}",
        f"training.checkpoint_every={STEPS + 1}",
        "logging.log_every=1",
    ]
    config = load_config(config_path, overrides=overrides)
    trainer = Trainer(config, wandb_enabled=False)
    losses = []
    trainer.logger.register_on_log_hook(
        lambda step, metrics: losses.append(metrics["train/loss"])
    )
    trainer.train()
    assert trainer.step == STEPS
    assert len(losses) == STEPS
    assert all(loss > 0 for loss in losses[1:])


def test_gpt2_dry_run():
    _require_tokenizer()
    _run_dry_run("configs/gpt2_124m.yaml")


def test_qwen3_dry_run():
    _require_tokenizer()
    _run_dry_run("configs/qwen3_51m.yaml")
