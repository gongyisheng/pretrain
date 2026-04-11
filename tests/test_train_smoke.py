"""Dry-run training tests for GPT-2 and Qwen3 using real configs and data."""
import pytest
import sys

sys.path.insert(0, ".")

from src.utils.config import load_config
from src.training.trainer import Trainer


STEPS = 1


@pytest.fixture(params=["torch", "triton"])
def backend(request):
    return request.param


def _run_dry_run(config_path, backend):
    overrides = [
        f"debug.max_steps={STEPS}",
        f"training.eval_every={STEPS + 1}",
        f"training.checkpoint_every={STEPS + 1}",
        f"training.backend={backend}",
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


def test_gpt2_dry_run(backend):
    _run_dry_run("configs/gpt2_124m.yaml", backend)


def test_qwen3_dry_run(backend):
    _run_dry_run("configs/qwen3_145m.yaml", backend)
