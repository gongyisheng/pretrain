import pytest
import os
import tempfile
import numpy as np
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from src.utils.config import TrainConfig, ModelConfig, DataConfig, TrainingConfig, OptimizerConfig, SchedulerConfig, LoggingConfig
from src.training.trainer import Trainer


def _make_tiny_tokenizer(path):
    """Create a minimal BPE tokenizer for testing."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<|endoftext|>"])
    tok.train_from_iterator(["hello world test data"] * 10, trainer=trainer)
    os.makedirs(path, exist_ok=True)
    tok.save(os.path.join(path, "tokenizer.json"))


def _tiny_config(tmp_dir):
    """Config for a tiny model that trains in seconds."""
    tokens = np.arange(4096, dtype=np.uint16)
    train_path = os.path.join(tmp_dir, "train.bin")
    val_path = os.path.join(tmp_dir, "val.bin")
    tokens.tofile(train_path)
    tokens[:512].tofile(val_path)

    tok_path = os.path.join(tmp_dir, "tokenizer")
    _make_tiny_tokenizer(tok_path)

    return TrainConfig(
        max_seq_len=64,
        model=ModelConfig(arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=4096),
        data=DataConfig(dataset="test", tokenizer_path=tok_path, data_dir=tmp_dir, val_split=0.01, num_workers=0),
        training=TrainingConfig(
            batch_size=4, gradient_accumulation_steps=1, max_steps=5,
            mixed_precision="no", activation_checkpointing=False,
            grad_clip=1.0, checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
            checkpoint_every=3, eval_every=3, eval_steps=2,
        ),
        optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.0, betas=[0.9, 0.95]),
        scheduler=SchedulerConfig(name="cosine", warmup_steps=1, min_lr=1e-4),
        logging=LoggingConfig(wandb_project="test", wandb_run_name="test", log_every=1),
    )


def test_trainer_runs_without_error():
    with tempfile.TemporaryDirectory() as tmp:
        config = _tiny_config(tmp)
        trainer = Trainer(config, wandb_enabled=False)
        trainer.train()
        assert trainer.step == 5


def test_trainer_saves_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        config = _tiny_config(tmp)
        trainer = Trainer(config, wandb_enabled=False)
        trainer.train()
        ckpt_dir = os.path.join(tmp, "ckpt")
        assert os.path.exists(os.path.join(ckpt_dir, "step_3.pt"))


def test_trainer_loss_decreases():
    with tempfile.TemporaryDirectory() as tmp:
        config = _tiny_config(tmp)
        config.training.max_steps = 20
        config.training.eval_every = 100
        config.training.checkpoint_every = 100
        trainer = Trainer(config, wandb_enabled=False)
        trainer.train()
        # loss_history[0] is always 0.0 due to deferred loss logging (prev_loss_tensor is None on step 1)
        assert trainer.metrics.loss_history[1] > trainer.metrics.loss_history[-1]


def _tiny_moe_config(tmp_dir):
    """Config for a tiny MoE model that trains in seconds."""
    tokens = np.arange(4096, dtype=np.uint16)
    train_path = os.path.join(tmp_dir, "train.bin")
    val_path = os.path.join(tmp_dir, "val.bin")
    tokens.tofile(train_path)
    tokens[:512].tofile(val_path)

    return TrainConfig(
        max_seq_len=64,
        model=ModelConfig(
            arch="qwen3_moe",
            n_layers=2,
            n_heads=2,
            n_kv_heads=2,
            d_model=64,
            intermediate_size=32,
            vocab_size=4096,
            n_experts=4,
            n_experts_per_token=2,
            moe_aux_loss_coef=0.01,
        ),
        data=DataConfig(dataset="test", tokenizer_path="", data_dir=tmp_dir, val_split=0.01, num_workers=0),
        training=TrainingConfig(
            batch_size=4, gradient_accumulation_steps=1, max_steps=5,
            mixed_precision="no", activation_checkpointing=False,
            grad_clip=1.0, checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
            checkpoint_every=3, eval_every=3, eval_steps=2,
        ),
        optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.0, betas=[0.9, 0.95]),
        scheduler=SchedulerConfig(name="cosine", warmup_steps=1, min_lr=1e-4),
        logging=LoggingConfig(wandb_project="test", wandb_run_name="test", log_every=1),
    )


def test_trainer_moe_runs_without_error():
    with tempfile.TemporaryDirectory() as tmp:
        config = _tiny_moe_config(tmp)
        trainer = Trainer(config, wandb_enabled=False)
        trainer.train()
        assert trainer.step == 5
