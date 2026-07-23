"""Unit tests for Trainer.

I/O is mocked: np.memmap is patched so PretrainDataset reads from in-memory
arrays instead of real .bin files, and tokenizer_path is empty so no BPE
training happens. Checkpoint writes are kept real (that's what we're testing).
"""

import os
import tempfile

import numpy as np
import pytest

from src.training.trainer import Trainer
from src.utils.config import (
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingConfig,
)


@pytest.fixture
def mock_memmap(monkeypatch):
    """Replace np.memmap with a dict-backed lookup keyed by path string."""
    storage: dict[str, np.ndarray] = {}

    def fake_memmap(path, dtype, mode):
        return storage[path]

    monkeypatch.setattr(np, "memmap", fake_memmap)
    return storage


def _seed_data(mock_memmap, tmp_dir):
    """Register fake train.bin / val.bin in the memmap mock."""
    tokens = np.arange(4096, dtype=np.uint16)
    mock_memmap[os.path.join(tmp_dir, "train.bin")] = tokens
    mock_memmap[os.path.join(tmp_dir, "val.bin")] = tokens[:512]


def _tiny_config(tmp_dir):
    """Tiny GPT-2 trainer config; tokenizer_path empty so no tokenizer is loaded."""
    return TrainConfig(
        max_seq_len=64,
        model=ModelConfig(
            n_layers=2,
            d_model=64,
            vocab_size=4096,
            attn=[{"attn_cls": "mha", "attn_kwargs": {"n_heads": 2, "bias": True}}],
            mlp=[
                {
                    "mlp_cls": "dense",
                    "mlp_kwargs": {"activation": "gelu", "gated": False, "bias": True},
                }
            ],
            norm_cls="layernorm",
            pos_emb_cls="learned",
        ),
        data=DataConfig(
            dataset="test",
            tokenizer_path="",
            data_dir=tmp_dir,
            val_split=0.01,
            num_workers=0,
        ),
        training=TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=1,
            max_steps=5,
            mixed_precision="no",
            activation_checkpointing=False,
            grad_clip=1.0,
            checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
            checkpoint_every=3,
            eval_every=3,
            eval_steps=2,
        ),
        optimizer=OptimizerConfig(
            name="adamw", lr=1e-3, weight_decay=0.0, betas=[0.9, 0.95]
        ),
        scheduler=SchedulerConfig(name="cosine", warmup_steps=1, min_lr=1e-4),
        logging=LoggingConfig(wandb_project="test", wandb_run_name="test", log_every=1),
    )


def _tiny_moe_config(tmp_dir):
    """Tiny qwen3_moe trainer config; tokenizer_path empty so no tokenizer is loaded."""
    return TrainConfig(
        max_seq_len=64,
        model=ModelConfig(
            n_layers=2,
            d_model=64,
            vocab_size=4096,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 2, "n_kv_heads": 2}}],
            mlp=[
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "intermediate_size": 32,
                        "n_routed_experts": 4,
                        "n_routed_experts_per_token": 2,
                        "aux_loss": True,
                        "aux_loss_coef": 1e-3,
                    },
                }
            ],
        ),
        data=DataConfig(
            dataset="test",
            tokenizer_path="",
            data_dir=tmp_dir,
            val_split=0.01,
            num_workers=0,
        ),
        training=TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=1,
            max_steps=5,
            # Dropless MoE requires bf16 (torch._grouped_mm is bf16-only under compile).
            mixed_precision="bf16",
            activation_checkpointing=False,
            grad_clip=1.0,
            checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
            checkpoint_every=3,
            eval_every=3,
            eval_steps=2,
        ),
        optimizer=OptimizerConfig(
            name="adamw", lr=1e-3, weight_decay=0.0, betas=[0.9, 0.95]
        ),
        scheduler=SchedulerConfig(name="cosine", warmup_steps=1, min_lr=1e-4),
        logging=LoggingConfig(wandb_project="test", wandb_run_name="test", log_every=1),
    )


def test_trainer_runs_without_error(mock_memmap):
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        trainer = Trainer(_tiny_config(tmp), wandb_enabled=False)
        trainer.train()
        assert trainer.step == 5


def test_trainer_saves_checkpoint(mock_memmap):
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        trainer = Trainer(_tiny_config(tmp), wandb_enabled=False)
        trainer.train()
        ckpt_dir = os.path.join(tmp, "ckpt")
        assert os.path.exists(os.path.join(ckpt_dir, "step_3.pt"))


def test_trainer_loss_decreases(mock_memmap):
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        config = _tiny_config(tmp)
        config.training.max_steps = 20
        config.training.eval_every = 100
        config.training.checkpoint_every = 100
        trainer = Trainer(config, wandb_enabled=False)
        losses = []
        trainer.logger.register_on_log_hook(
            lambda step, metrics: losses.append(metrics["train/loss"])
        )
        trainer.train()
        # losses[0] is always 0.0 due to deferred loss logging (prev_loss_tensor is None on step 1)
        assert losses[1] > losses[-1]


def test_trainer_moe_runs_without_error(mock_memmap):
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        trainer = Trainer(_tiny_moe_config(tmp), wandb_enabled=False)
        trainer.train()
        assert trainer.step == 5


def test_trainer_rejects_unknown_loss_fn(mock_memmap):
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        cfg = _tiny_config(tmp)
        cfg.training.loss_fn = "not_a_real_loss"
        with pytest.raises(ValueError, match="unknown loss_fn"):
            Trainer(cfg, wandb_enabled=False)
