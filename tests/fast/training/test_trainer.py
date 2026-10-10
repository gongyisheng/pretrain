import os
import tempfile

import numpy as np
import pytest

from src.quant.linear import QuantizedLinear
from src.training.trainer import Trainer
from src.utils.config import (
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    QuantizationConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingConfig,
)
from tests.fast.helper import cuda_only

pytestmark = cuda_only


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
# quant metrics wiring: flag off -> no hooks, no train-quant/ keys;
# flag on -> train-quant/ keys dispatched.
# ---------------------------------------------------------------------------


def test_quant_metrics_without_quantization(mock_memmap):
    """With quantization disabled, the install pass finds no quantized sites, so
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


def _tiny_quant_config(
    tmp_dir,
    enabled_after_steps=0,
    enabled_before_steps=None,
):
    """Tiny CUDA trainer with quantization and quantization metrics enabled."""
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
    )
    cfg.quantization = QuantizationConfig(
        enabled=True,
        enabled_after_steps=enabled_after_steps,
        enabled_before_steps=enabled_before_steps,
        dtype={"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
    )
    cfg.logging.log_quant_metrics = True
    cfg.logging.log_every = 1
    return cfg


QUANT_METRIC_AFTER_STEPS = [0, 1]
QUANT_METRIC_BEFORE_STEPS = [None, 0, 1]


@pytest.mark.parametrize("enabled_after_steps", QUANT_METRIC_AFTER_STEPS)
@pytest.mark.parametrize("enabled_before_steps", QUANT_METRIC_BEFORE_STEPS)
def test_quant_metrics_enabled_dispatches_quant_keys(
    mock_memmap, enabled_after_steps, enabled_before_steps
):
    """Quantization metrics appear only during active training steps."""
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        cfg = _tiny_quant_config(tmp, enabled_after_steps, enabled_before_steps)

        trainer = Trainer(cfg, wandb_enabled=False)
        logged = []
        trainer.logger.register_on_log_hook(
            lambda step, metrics: logged.append(metrics)
        )
        trainer.train()
        assert [
            any(key.startswith("train-quant/") for key in metrics) for metrics in logged
        ] == [
            enabled_after_steps <= step
            and (enabled_before_steps is None or step < enabled_before_steps)
            for step in range(cfg.training.max_steps)
        ]


QUANTIZATION_AFTER_STEPS = [0, 1, 3]
QUANTIZATION_BEFORE_STEPS = [None, 0, 2, 3]


@pytest.mark.parametrize("enabled_after_steps", QUANTIZATION_AFTER_STEPS)
@pytest.mark.parametrize("enabled_before_steps", QUANTIZATION_BEFORE_STEPS)
def test_trainer_train_quantization_window(
    mock_memmap, enabled_after_steps, enabled_before_steps
):
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        cfg = _tiny_quant_config(
            tmp,
            enabled_after_steps=enabled_after_steps,
            enabled_before_steps=enabled_before_steps,
        )
        cfg.training.max_steps = 4
        cfg.training.gradient_accumulation_steps = 2
        trainer = Trainer(cfg, wandb_enabled=False)
        modules = [
            module
            for module in trainer.eager_model.modules()
            if isinstance(module, QuantizedLinear)
        ]
        assert modules and all(not module.quantization_enabled for module in modules)
        phases = []
        modules[0].register_forward_pre_hook(
            lambda quantized_module, inputs: phases.append(
                tuple(module.quantization_enabled for module in modules)
            )
        )
        trainer.train()
        assert phases == [
            (
                enabled_after_steps <= step
                and (enabled_before_steps is None or step < enabled_before_steps),
            )
            * len(modules)
            for step in range(cfg.training.max_steps)
            for _ in range(cfg.training.gradient_accumulation_steps)
        ]


RESUME_AFTER_STEPS = [1, 3]
RESUME_BEFORE_STEPS = [None, 3]
RESUME_STEPS = [2, 3, 4]


@pytest.mark.parametrize("enabled_after_steps", RESUME_AFTER_STEPS)
@pytest.mark.parametrize("enabled_before_steps", RESUME_BEFORE_STEPS)
@pytest.mark.parametrize("step", RESUME_STEPS)
def test_trainer_resume_quantization_window(
    mock_memmap, enabled_after_steps, enabled_before_steps, step
):
    with tempfile.TemporaryDirectory() as tmp:
        _seed_data(mock_memmap, tmp)
        cfg = _tiny_quant_config(
            tmp,
            enabled_after_steps=enabled_after_steps,
            enabled_before_steps=enabled_before_steps,
        )
        cfg.training.max_steps = step + 1
        cfg.training.early_stop = step
        cfg.training.checkpoint_every = step
        trainer = Trainer(cfg, wandb_enabled=False)
        trainer.train()
        checkpoint = os.path.join(cfg.training.checkpoint_dir, f"step_{step}.pt")
        resumed = Trainer(cfg, wandb_enabled=False, resume_from=checkpoint)
        modules = [
            module
            for module in resumed.eager_model.modules()
            if isinstance(module, QuantizedLinear)
        ]
        assert modules and all(
            module.quantization_enabled
            is (
                enabled_after_steps <= step
                and (enabled_before_steps is None or step < enabled_before_steps)
            )
            for module in modules
        )
        cfg.training.early_stop = step + 1
        resumed.train()
        assert resumed.step == step + 1


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
            cfg = _tiny_quant_config(tmp)
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
