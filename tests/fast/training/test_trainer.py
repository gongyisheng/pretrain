"""Unit tests for Trainer.

I/O is mocked: np.memmap is patched so PretrainDataset reads from in-memory
arrays instead of real .bin files, and tokenizer_path is empty so no BPE
training happens. Checkpoint writes are kept real (that's what we're testing).
"""

import os
import tempfile

import numpy as np
import pytest
import torch

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

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="quantized trainer test needs CUDA"
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
                    "mlp_kwargs": {
                        "activation_cls": "gelu",
                        "bias": True,
                    },
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
            grad_clip=1.0,
            checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
            checkpoint_every=3,
            eval_every=3,
            eval_steps=2,
        ),
        optimizer=OptimizerConfig("adamw", lr=1e-3, weight_decay=0.0),
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
            # Dropless MoE uses the Triton grouped GEMM reduced-precision path.
            mixed_precision="bf16",
            grad_clip=1.0,
            checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
            checkpoint_every=3,
            eval_every=3,
            eval_steps=2,
        ),
        optimizer=OptimizerConfig("adamw", lr=1e-3, weight_decay=0.0),
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


# ---------------------------------------------------------------------------
# quant metrics wiring: flag off -> no hooks, no train-quant/ keys (runs everywhere);
# flag on -> train-quant/ keys dispatched (needs CUDA, see cuda_only below).
# ---------------------------------------------------------------------------


def test_quant_metrics_without_a_quant_recipe(mock_memmap):
    """With no quantization recipe, the install pass finds no quantized sites, so
    there is no accumulator to fold into and no train-quant/ key is dispatched --
    independent of log_quant_metrics, which now defaults to True."""
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        trainer = Trainer(_tiny_config(tmp), wandb_enabled=False)
        # the install pass never ran, so no accumulator exists to fold into
        assert not any(
            type(m).__name__ == "QuantizationStats" for m in trainer.model.modules()
        )
        logged = []
        trainer.logger.register_on_log_hook(
            lambda step, metrics: logged.append(metrics)
        )
        trainer.train()
        assert not any(
            k.startswith("train-quant/") for metrics in logged for k in metrics
        )


def _tiny_fp8_config(tmp_dir):
    """_tiny_config, but on cuda with a tensorwise fp8 quant rule (mirrors
    tests/fast/quant/test_converter.py's `_cfg`) and quant metrics on."""
    cfg = _tiny_config(tmp_dir)
    cfg.training = TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=2,
        device="cuda",
        mixed_precision="bf16",
        grad_clip=1.0,
        checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
        checkpoint_every=100,
        eval_every=100,
        eval_steps=2,
        enable_torch_compile=False,
        quantization={"enabled": True, "dtype": {"recipe": "fp8"}},
    )
    cfg.logging.log_quant_metrics = True
    cfg.logging.log_every = 1
    return cfg


@cuda_only
def test_quant_metrics_enabled_dispatches_quant_keys(mock_memmap):
    """log_quant_metrics=True with an fp8 quant recipe: at least one train-quant/
    key from the diagnostic pass reaches the logger."""
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        cfg = _tiny_fp8_config(tmp)

        trainer = Trainer(cfg, wandb_enabled=False)
        logged = []
        trainer.logger.register_on_log_hook(
            lambda step, metrics: logged.append(metrics)
        )
        trainer.train()
        assert any(k.startswith("train-quant/") for metrics in logged for k in metrics)


def test_activation_norms_reach_both_train_and_val_keys(mock_memmap):
    """One run, both windows: the train window opens on a log step, the val window
    on every eval."""
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        cfg = _tiny_config(tmp)
        cfg.logging.log_activation_norms = True
        cfg.logging.log_every = 1
        cfg.training.eval_every = 1

        trainer = Trainer(cfg, wandb_enabled=False)
        logged = []
        trainer.logger.register_on_log_hook(
            lambda step, metrics: logged.append(metrics)
        )
        trainer.train()

        assert any(k.startswith("train-act/norm/") for m in logged for k in m)
        assert any(k.startswith("val-act/norm/") for m in logged for k in m)


@cuda_only
def test_quant_diagnostics_do_not_change_training(mock_memmap):
    """The diagnostic pass runs its own fwd/bwd; the trained loss must not move.

    Dropout is on so the training forward consumes RNG — without the diagnostic's
    RNG restore its own forward advances the stream and the next step's loss moves.
    Needs >2 steps: loss logging is deferred one step (losses[0] is always 0.0), so
    a run of 2 only ever reports step 1's loss, which precedes every diagnostic.
    """
    losses = {}
    for log_quant_metrics in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_data(mock_memmap, tmp)
            cfg = _tiny_fp8_config(tmp)
            cfg.model.dropout_embd = 0.1
            cfg.training.max_steps = 4
            cfg.logging.log_quant_metrics = log_quant_metrics
            cfg.logging.log_every = 1
            trainer = Trainer(cfg, wandb_enabled=False)
            logged = []
            trainer.logger.register_on_log_hook(
                lambda step, metrics: logged.append(metrics)
            )
            trainer.train()
            losses[log_quant_metrics] = [
                m["train/loss"] for m in logged if "train/loss" in m
            ]
    # guard the comparison: 4 entries, and the post-diagnostic ones are real losses
    assert len(losses[False]) == cfg.training.max_steps
    assert all(loss > 0 for loss in losses[False][2:])
    assert losses[True] == losses[False]
