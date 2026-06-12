"""Tests for MetricsTracker — the stateful assembly layer.

Pure metric math is tested in tests/fast/utils/test_metric_utils.py. Here we
exercise the tracker's state machine: per-step accumulation, the log-cadence
gate, step-norm gating, and eval accumulation. A FakeLogger captures dispatched
dicts so tests assert on what would be logged without touching W&B.
"""

import math

import pytest
import torch

from src.model.registry import build_model
from src.training.metrics import MetricsTracker
from src.utils.config import ModelConfig, TrainConfig


class FakeLogger:
    """Captures (step, dict) / (key, text) calls in place of WandbLogger."""

    def __init__(self):
        self.logs: list[tuple[int, dict]] = []
        self.texts: list[tuple[str, str, int]] = []

    def log(self, metrics: dict, step: int):
        self.logs.append((step, metrics))

    def log_text(self, key: str, text: str, step: int):
        self.texts.append((key, text, step))

    def finish(self):
        pass


def _cfg(task="pretrain", log_every=2, **logging):
    cfg = TrainConfig(
        model=ModelConfig(
            n_layers=2,
            d_model=64,
            vocab_size=256,
            attn_cls="mha",
            attn_kwargs={"n_heads": 2},
            mlp_cls="dense",
            mlp_kwargs={"gated": False, "bias": True, "activation": "gelu"},
        )
    )
    cfg.task = task
    cfg.logging.log_every = log_every
    for k, v in logging.items():
        setattr(cfg.logging, k, v)
    return cfg


def _tracker(cfg=None):
    cfg = cfg or _cfg()
    return MetricsTracker(cfg, device="cpu", logger=FakeLogger()), cfg


def _linear_setup():
    """A tiny model + optimizer + disabled scaler with grads populated."""
    model = torch.nn.Linear(8, 256)
    model(torch.randn(4, 8)).sum().backward()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=False)
    return model, opt, scaler


def _do_step(tracker, model, opt, scaler, step, *, loss=2.0, grad_norm=0.5):
    """Run one tracker step (snapshot → optimizer step → on_step)."""
    tracker.snapshot_pre_step(model, step)
    scale_before = scaler.get_scale()
    opt.step()
    tracker.on_step(
        loss=loss,
        grad_norm=grad_norm,
        model=model,
        optimizer=opt,
        scaler=scaler,
        scale_before=scale_before,
    )


# ---------------------------------------------------------------------------
# log_train: cadence gate, assembly, dispatch, reset
# ---------------------------------------------------------------------------


def test_log_train_returns_none_off_cadence():
    tracker, _ = _tracker(_cfg(log_every=10))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    assert tracker.log_train(step=5, model=model, optimizer=opt) is None
    assert tracker.logger.logs == []  # nothing dispatched off cadence


def test_log_train_assembles_and_dispatches_on_cadence():
    tracker, _ = _tracker(_cfg(log_every=2))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=1, loss=2.0, grad_norm=0.5)
    d = tracker.log_train(step=2, model=model, optimizer=opt)
    assert d is not None
    assert d["train/loss"] == 2.0
    assert d["grad_norm/total"] == 0.5
    assert d["optim/lr"] == pytest.approx(3e-4)  # read from optimizer
    assert "perf/tokens_per_sec" in d and "perf/step_time_ms" in d
    assert "train/perplexity" in d  # pretrain
    # dispatched exactly once, at the given step
    assert len(tracker.logger.logs) == 1 and tracker.logger.logs[0][0] == 2


def test_log_train_resets_window_counters():
    tracker, _ = _tracker(_cfg(log_every=1))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0, grad_norm=99.0)  # exceeds grad_clip
    assert tracker._steps_since_log == 1
    tracker.log_train(step=1, model=model, optimizer=opt)
    assert tracker._steps_since_log == 0
    assert tracker._tokens_since_log == 0
    assert tracker._grad_clip_steps == 0


def test_grad_clip_ratio_reflects_window():
    tracker, _ = _tracker(_cfg(log_every=2))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0, grad_norm=99.0)  # > grad_clip (1.0)
    _do_step(tracker, model, opt, scaler, step=1, grad_norm=0.1)  # within clip
    d = tracker.log_train(step=2, model=model, optimizer=opt)
    assert d["optim/grad_clip_ratio"] == pytest.approx(0.5)  # 1 of 2 steps clipped


# ---------------------------------------------------------------------------
# total_tokens / tokens_per_step
# ---------------------------------------------------------------------------


def test_tokens_per_step_from_config_and_total_accumulates():
    cfg = _cfg(log_every=2)
    cfg.training.batch_size = 4
    cfg.training.gradient_accumulation_steps = 3
    cfg.max_seq_len = 16
    tracker, _ = _tracker(cfg)
    assert tracker.tokens_per_step == 4 * 3 * 16
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    for s in range(2):
        _do_step(tracker, model, opt, scaler, step=s)
    assert tracker.total_tokens == 2 * tracker.tokens_per_step
    d = tracker.log_train(step=2, model=model, optimizer=opt)
    assert d["train/total_tokens"] == tracker.total_tokens
    assert d["train/flops"] == tracker.flops_per_token["total"] * tracker.total_tokens


# ---------------------------------------------------------------------------
# optimizer step-norm gating
# ---------------------------------------------------------------------------


def test_step_norms_present_on_log_step():
    tracker, _ = _tracker(_cfg(log_every=2, log_optimizer_step_norms=True))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    # step 1 -> (1+1)%2==0, so snapshot fires and on_step computes the norms
    _do_step(tracker, model, opt, scaler, step=1)
    d = tracker.log_train(step=2, model=model, optimizer=opt)
    assert "optim/param_step_norm" in d
    assert "optim/momentum_norm" in d
    assert "optim/variance_norm" in d  # AdamW has a second moment


def test_step_norms_gated_off_non_log_steps():
    """snapshot_pre_step is a no-op except on the pre-log step."""
    tracker, _ = _tracker(_cfg(log_every=10, log_optimizer_step_norms=True))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    for s in range(9):  # steps 0..8: (s+1)%10 != 0
        tracker.snapshot_pre_step(model, s)
        assert tracker._param_snapshot is None
    tracker.snapshot_pre_step(model, 9)  # (9+1)%10 == 0
    assert tracker._param_snapshot is not None


def test_step_norms_absent_when_disabled():
    tracker, _ = _tracker(_cfg(log_every=1, log_optimizer_step_norms=False))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    assert tracker._param_snapshot is None
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert "optim/param_step_norm" not in d
    assert "optim/momentum_norm" not in d


def test_non_finite_loss_not_guarded_here():
    """The tracker no longer guards loss finiteness — it just caches it.
    (The trainer owns the RuntimeError guard.)"""
    tracker, _ = _tracker(_cfg(log_every=1))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0, loss=float("nan"))
    assert math.isnan(tracker._last_loss)


# ---------------------------------------------------------------------------
# Per-layer grad norms get the grad_norm/ prefix in the dict
# ---------------------------------------------------------------------------


def test_layer_grad_norms_prefixed_in_log_dict():
    tracker, _ = _tracker(_cfg(log_every=1, log_layer_grad_norms=True))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert "grad_norm/weight" in d and "grad_norm/bias" in d
    assert "grad_norm/total" in d  # the scalar total stays


# ---------------------------------------------------------------------------
# Eval: accumulate then finalize
# ---------------------------------------------------------------------------


def test_eval_pretrain_loss_and_perplexity():
    tracker, _ = _tracker(_cfg(task="pretrain"))
    tracker.eval_begin()
    logits = torch.randn(2, 5, 256)
    labels = torch.randint(0, 256, (2, 5))
    tracker.eval_step(loss=1.5, logits=logits, labels=labels)
    tracker.eval_step(loss=2.5, logits=logits, labels=labels)
    d = tracker.log_eval(step=10)
    assert d["val/loss"] == pytest.approx(2.0)  # mean of 1.5, 2.5
    assert d["val/perplexity"] == pytest.approx(math.exp(2.0))
    assert "val/bpb" not in d  # no tokenizer passed
    assert tracker.logger.logs[-1][0] == 10


def test_eval_pretrain_bpb_with_fake_tokenizer():
    tracker, _ = _tracker(_cfg(task="pretrain"))

    class FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)  # 1 byte per token

    tracker.eval_begin()
    labels = torch.tensor([[1, 2, 3, -100]])  # 3 loss-contributing tokens
    logits = torch.randn(1, 4, 256)
    tracker.eval_step(loss=2.0, logits=logits, labels=labels, tokenizer=FakeTok())
    d = tracker.log_eval(step=1)
    # 3 tokens / 3 bytes -> tokens_per_byte = 1.0 ; bpb = loss * 1 / ln2
    assert d["val/bpb"] == pytest.approx(2.0 / math.log(2))


def test_eval_sft_accuracy_excludes_eot():
    tracker, _ = _tracker(_cfg(task="sft"))
    tracker.eval_begin()
    logits = torch.zeros(1, 3, 4)
    logits[0, torch.arange(3), torch.tensor([1, 2, 3])] = 10.0  # preds 1, 2, 3
    labels = torch.tensor([[1, 2, 3]])
    tracker.eval_step(loss=0.1, logits=logits, labels=labels, eot_token_id=0)
    d = tracker.log_eval(step=1)
    assert d["val/val_acc"] == pytest.approx(1.0)
    assert "val/perplexity" not in d and "val/bpb" not in d


def test_eval_sft_train_acc_passthrough():
    tracker, _ = _tracker(_cfg(task="sft"))
    tracker.eval_begin()
    logits = torch.zeros(1, 2, 4)
    logits[0, torch.arange(2), torch.tensor([1, 2])] = 10.0
    tracker.eval_step(loss=0.1, logits=logits, labels=torch.tensor([[1, 2]]))
    d = tracker.log_eval(step=1, train_avg_acc=0.99)
    assert d["val/train_acc"] == pytest.approx(0.99)


def test_eval_moe_aux_loss_subtracts_floor():
    cfg = TrainConfig(
        model=ModelConfig(
            n_layers=2,
            d_model=64,
            vocab_size=256,
            attn_cls="gqa",
            attn_kwargs={"n_heads": 2, "n_kv_heads": 1, "qk_norm": True},
            mlp_cls="moe",
            mlp_kwargs={
                "n_experts": 4,
                "n_experts_per_token": 2,
                "intermediate_size": 128,
            },
        )
    )
    cfg.task = "pretrain"
    tracker = MetricsTracker(cfg, device="cpu", logger=FakeLogger())
    floor = tracker._aux_floor
    tracker.eval_begin()
    logits = torch.randn(1, 4, 256)
    labels = torch.randint(0, 256, (1, 4))
    tracker.eval_step(loss=2.0, logits=logits, labels=labels, aux_loss=floor + 0.5)
    d = tracker.log_eval(step=1)
    assert d["val/aux_loss"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# print_model_summary
# ---------------------------------------------------------------------------


def test_print_model_summary_dense(capsys):
    cfg = TrainConfig(max_seq_len=128, model=_cfg().model)
    tracker = MetricsTracker(cfg, device="cpu", logger=FakeLogger())
    model = build_model(cfg)
    assert tracker.print_model_summary(model) is None
    out = capsys.readouterr().out
    assert "Model: mha+dense" in out and "non-embedding" in out and "device=cpu" in out


def test_print_model_summary_moe(capsys):
    cfg = TrainConfig(
        max_seq_len=128,
        model=ModelConfig(
            n_layers=2,
            d_model=64,
            vocab_size=256,
            attn_cls="gqa",
            attn_kwargs={"n_heads": 2, "n_kv_heads": 1, "qk_norm": True},
            mlp_cls="moe",
            mlp_kwargs={
                "n_experts": 4,
                "n_experts_per_token": 2,
                "intermediate_size": 128,
            },
        ),
    )
    tracker = MetricsTracker(cfg, device="cpu", logger=FakeLogger())
    tracker.print_model_summary(build_model(cfg))
    out = capsys.readouterr().out
    assert "total params" in out and "active non-embedding" in out
