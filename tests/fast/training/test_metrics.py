"""Tests for MetricsCollector — the stateful assembly layer.

Pure metric math is tested in tests/fast/utils/test_metric_utils.py. Here we
exercise the tracker's state machine: per-step accumulation, the log-cadence
gate, step-norm gating, and eval accumulation. A FakeLogger captures dispatched
dicts so tests assert on what would be logged without touching W&B.
"""

import math

import pytest
import torch

from src.layers.block import TransformerBlock
from src.model import build_model
from src.training.metrics import (
    MetricsCollector,
    apply_activation_monitoring,
    reset_activation_stats,
    set_activation_monitoring_status,
)
from src.utils.config import ModelConfig, TrainConfig
from src.utils.metric_utils import compute_activation_norm
from tests.fast.helpers import make_attn_mask


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
            attn=[{"attn_cls": "mha", "attn_kwargs": {"n_heads": 2}}],
            mlp=[
                {
                    "mlp_cls": "dense",
                    "mlp_kwargs": {"gated": False, "bias": True, "activation": "gelu"},
                }
            ],
        )
    )
    cfg.task = task
    cfg.logging.log_every = log_every
    for k, v in logging.items():
        setattr(cfg.logging, k, v)
    return cfg


def _tracker(cfg=None):
    cfg = cfg or _cfg()
    return MetricsCollector(cfg, device="cpu", logger=FakeLogger()), cfg


def _linear_setup():
    """A tiny model + optimizer + disabled scaler with grads populated."""
    model = torch.nn.Linear(8, 256)
    model(torch.randn(4, 8)).sum().backward()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=False)
    return model, opt, scaler


def _do_step(tracker, model, opt, scaler, step, *, loss=2.0, grad_norm=0.5):
    """Run one tracker step (snapshot → optimizer step → on_train_step)."""
    tracker.snapshot_pre_step(model, step)
    scale_before = scaler.get_scale()
    opt.step()
    tracker.on_train_step(
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
    assert d["grad/norm/total"] == 0.5
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
# quant metrics flush merge
# ---------------------------------------------------------------------------


def test_log_train_merges_quant_metrics():
    tracker, _ = _tracker(_cfg(log_every=1))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(
        step=1,
        model=model,
        optimizer=opt,
        quant_metrics={"quant/sqnr/act/layer_0": 42.0},
    )
    assert d["quant/sqnr/act/layer_0"] == 42.0
    assert tracker.logger.logs[-1][1]["quant/sqnr/act/layer_0"] == 42.0


def test_log_train_without_quant_metrics_has_no_quant_keys():
    tracker, _ = _tracker(_cfg(log_every=1))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert not any(k.startswith("quant/") for k in d)


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
    assert d["train/flops"] == tracker._flops_per_token * tracker.total_tokens


# ---------------------------------------------------------------------------
# optimizer step-norm gating
# ---------------------------------------------------------------------------


def test_step_norms_present_on_log_step():
    tracker, _ = _tracker(_cfg(log_every=2, log_optimizer_step_norms=True))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    # step 1 -> (1+1)%2==0, so snapshot fires and on_train_step computes the norms
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
# Per-layer grad norms get the grad/norm/ prefix in the dict
# ---------------------------------------------------------------------------


def test_grad_norms_prefixed_in_log_dict():
    tracker, _ = _tracker(_cfg(log_every=1, log_grad_norms=True))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert "grad/norm/weight" in d and "grad/norm/bias" in d
    assert "grad/norm/total" in d  # the scalar total stays


def test_weight_norms_prefixed_in_log_dict():
    tracker, _ = _tracker(_cfg(log_every=1, log_weight_norms=True))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert "weight/norm/weight" in d and "weight/norm/bias" in d


def test_grad_svd_metrics_prefixed_in_log_dict():
    """Flag unset → on by default."""
    tracker, _ = _tracker(_cfg(log_every=1))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert "grad/srank/weight" in d and "grad/pr/weight" in d
    assert not any(k.startswith("grad/srank/bias") for k in d)  # 1D params skipped


def test_grad_svd_metrics_not_logged_when_flag_off():
    tracker, _ = _tracker(_cfg(log_every=1, log_grad_svd_metrics=False))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert not any(k.startswith("grad/srank/") or k.startswith("grad/pr/") for k in d)


def test_weight_norms_absent_when_disabled():
    tracker, _ = _tracker(_cfg(log_every=1, log_weight_norms=False))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert not any(k.startswith("weight/norm/") for k in d)


# ---------------------------------------------------------------------------
# Eval: accumulate then finalize
# ---------------------------------------------------------------------------


def test_eval_pretrain_loss_and_perplexity():
    tracker, _ = _tracker(_cfg(task="pretrain"))
    tracker.eval_begin()
    logits = torch.randn(2, 5, 256)
    labels = torch.randint(0, 256, (2, 5))
    tracker.on_eval_step(loss=1.5, logits=logits, labels=labels)
    tracker.on_eval_step(loss=2.5, logits=logits, labels=labels)
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
    tracker.on_eval_step(loss=2.0, logits=logits, labels=labels, tokenizer=FakeTok())
    d = tracker.log_eval(step=1)
    # 3 tokens / 3 bytes -> tokens_per_byte = 1.0 ; bpb = loss * 1 / ln2
    assert d["val/bpb"] == pytest.approx(2.0 / math.log(2))


def test_eval_sft_accuracy_excludes_eot():
    tracker, _ = _tracker(_cfg(task="sft"))
    tracker.eval_begin()
    logits = torch.zeros(1, 3, 4)
    logits[0, torch.arange(3), torch.tensor([1, 2, 3])] = 10.0  # preds 1, 2, 3
    labels = torch.tensor([[1, 2, 3]])
    tracker.on_eval_step(loss=0.1, logits=logits, labels=labels, eot_token_id=0)
    d = tracker.log_eval(step=1)
    assert d["val/val_acc"] == pytest.approx(1.0)
    assert "val/perplexity" not in d and "val/bpb" not in d


def test_eval_sft_train_acc_passthrough():
    tracker, _ = _tracker(_cfg(task="sft"))
    tracker.eval_begin()
    logits = torch.zeros(1, 2, 4)
    logits[0, torch.arange(2), torch.tensor([1, 2])] = 10.0
    tracker.on_eval_step(loss=0.1, logits=logits, labels=torch.tensor([[1, 2]]))
    d = tracker.log_eval(step=1, train_avg_acc=0.99)
    assert d["val/train_acc"] == pytest.approx(0.99)


def test_eval_moe_aux_loss_subtracts_floor():
    cfg = TrainConfig(
        model=ModelConfig(
            n_layers=2,
            d_model=64,
            vocab_size=256,
            attn=[
                {
                    "attn_cls": "gqa",
                    "attn_kwargs": {"n_heads": 2, "n_kv_heads": 1, "qk_norm": True},
                }
            ],
            mlp=[
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "n_routed_experts": 4,
                        "aux_loss": True,
                        "aux_loss_coef": 1e-3,
                        "n_routed_experts_per_token": 2,
                        "intermediate_size": 128,
                    },
                }
            ],
        )
    )
    cfg.task = "pretrain"
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    model = build_model(cfg)
    model.eval()
    floor = tracker._aux_floor
    tracker.eval_begin()
    logits = torch.randn(1, 4, 256)
    labels = torch.randint(0, 256, (1, 4))
    tracker.on_eval_step(
        loss=2.0, logits=logits, labels=labels, model=model, aux_loss=floor + 0.5
    )
    d = tracker.log_eval(step=1)
    assert d["val/aux_loss"] == pytest.approx(0.5)


def test_aux_floor_is_coef_weighted():
    cfg = TrainConfig(
        model=ModelConfig(
            d_model=64,
            n_layers=2,
            vocab_size=256,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4, "n_kv_heads": 2}}],
            mlp=[
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "n_routed_experts": 4,
                        "n_routed_experts_per_token": 2,
                        "aux_loss": True,
                        "aux_loss_coef": 1e-3,
                    },
                    "layer_idx": [0],
                },
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "n_routed_experts": 4,
                        "n_routed_experts_per_token": 3,
                        "aux_loss": True,
                        "aux_loss_coef": 1e-2,
                    },
                },
            ],
        )
    )
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    # floor = 1e-3*2 + 1e-2*3 = 0.032
    assert abs(tracker._aux_floor - (1e-3 * 2 + 1e-2 * 3)) < 1e-9


def _moe_cfg():
    cfg = TrainConfig(
        model=ModelConfig(
            n_layers=2,
            d_model=64,
            vocab_size=256,
            attn=[
                {
                    "attn_cls": "gqa",
                    "attn_kwargs": {"n_heads": 2, "n_kv_heads": 1, "qk_norm": True},
                }
            ],
            mlp=[
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "n_routed_experts": 4,
                        "aux_loss": True,
                        "aux_loss_coef": 1e-3,
                        "n_routed_experts_per_token": 2,
                        "intermediate_size": 128,
                    },
                }
            ],
        )
    )
    cfg.task = "pretrain"
    return cfg


def _moe_blocks(model):
    from src.layers.mlp import SparseMoEBlock

    return [m for m in model.modules() if isinstance(m, SparseMoEBlock)]


def _moe_on_step(tracker, model, opt, step):
    """Drive one tracker step for an MoE model (no grads needed)."""
    scaler = torch.amp.GradScaler(enabled=False)
    tracker.snapshot_pre_step(model, step)
    tracker.on_train_step(
        loss=2.0,
        grad_norm=0.5,
        model=model,
        optimizer=opt,
        scaler=scaler,
        scale_before=scaler.get_scale(),
    )


def test_train_moe_maxvio_logged_per_layer():
    cfg = _moe_cfg()
    cfg.logging.log_every = 1
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    model = build_model(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    tracker.train_begin()
    blocks = _moe_blocks(model)
    blocks[0].expert_load.train_load.copy_(torch.tensor([10, 0, 5, 5]))
    blocks[1].expert_load.train_load.copy_(torch.tensor([4, 4, 4, 4]))
    _moe_on_step(tracker, model, opt, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)

    assert (
        "train-moe/maxvio_batch/layer_0" in d and "train-moe/maxvio_batch/layer_1" in d
    )
    assert "train-moe/maxvio_batch/mean" in d and "train-moe/maxvio_batch/max" in d
    assert d["train-moe/maxvio_batch/max"] >= d["train-moe/maxvio_batch/mean"] - 1e-9
    assert all(d[k] >= 0.0 for k in d if k.startswith("train-moe/maxvio_batch/"))


def test_train_moe_maxvio_uses_latest_step():
    # Train MaxVio reflects only the latest step's routing load (cached each
    # on_train_step, overwriting the prior). Step 0 stamps a balanced load and
    # step 1 a maximally-skewed one: the logged value tracks step 1 (maxvio 1.0),
    # confirming last-step semantics, not window accumulation.
    cfg = _moe_cfg()
    cfg.logging.log_every = 2
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    model = build_model(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    tracker.train_begin()
    block = _moe_blocks(model)[0]
    block.expert_load.train_load.copy_(torch.tensor([5, 5, 5, 5]))
    _moe_on_step(tracker, model, opt, step=0)
    block.expert_load.train_load.copy_(torch.tensor([10, 0, 5, 5]))
    _moe_on_step(tracker, model, opt, step=1)
    d = tracker.log_train(step=2, model=model, optimizer=opt)
    # latest load [10,0,5,5]: max=10, mean=5 -> (10-5)/5 = 1.0
    assert d["train-moe/maxvio_batch/layer_0"] == pytest.approx(1.0)


def test_eval_moe_maxvio_logs_both_variants():
    cfg = _moe_cfg()
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    model = build_model(cfg)
    model.eval()  # eval mode -> forward_meta returns eval_load
    blocks = _moe_blocks(model)
    logits = torch.randn(1, 4, 256)
    labels = torch.randint(0, 256, (1, 4))

    tracker.eval_begin()
    blocks[0].expert_load.eval_load.copy_(torch.tensor([10, 0, 5, 5]))
    blocks[1].expert_load.eval_load.copy_(torch.tensor([4, 4, 4, 4]))
    tracker.on_eval_step(loss=2.0, logits=logits, labels=labels, model=model)
    d = tracker.log_eval(step=1)

    for variant in ("maxvio_global", "maxvio_batch"):
        assert f"val-moe/{variant}/layer_0" in d and f"val-moe/{variant}/layer_1" in d
        assert f"val-moe/{variant}/mean" in d and f"val-moe/{variant}/max" in d
    assert all(v >= 0.0 for k, v in d.items() if k.startswith("val-moe/"))


def test_eval_moe_maxvio_global_vs_batch_semantics():
    # global sums load across batches then computes one MaxVio; batch computes
    # MaxVio per batch then averages. Batches [10,0,5,5] and [0,10,5,5] sum to
    # [10,10,10,10] -> global 0, but each batch has MaxVio 1.0 -> batch avg 1.0.
    cfg = _moe_cfg()
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    model = build_model(cfg)
    model.eval()  # eval mode -> forward_meta returns eval_load
    block = _moe_blocks(model)[0]
    logits = torch.randn(1, 4, 256)
    labels = torch.randint(0, 256, (1, 4))

    tracker.eval_begin()
    for counts in ([10, 0, 5, 5], [0, 10, 5, 5]):
        block.expert_load.eval_load.copy_(torch.tensor(counts))
        tracker.on_eval_step(loss=2.0, logits=logits, labels=labels, model=model)
    d = tracker.log_eval(step=1)

    assert d["val-moe/maxvio_global/layer_0"] == pytest.approx(0.0)
    assert d["val-moe/maxvio_batch/layer_0"] == pytest.approx(1.0)


def _dense_first_moe_rest_cfg():
    """4-layer dense-first mixed stack: layer 0 dense, layers 1-3 MoE (k=2),
    all 3 MoE layers using aux_loss.

    n_layers (4) != n_moe_layers (3) on purpose: this is the config that would
    catch a regression to the old `n_layers * k` aux-floor formula (8, wrong)
    vs. the correct per-layer sum (3 * 2 = 6). All MoE layers use aux_loss here
    (not expert_bias), so the aux-loss-layers-only floor and the all-MoE-layers
    floor coincide (6 either way) — see
    `test_metrics_tracker_aux_floor_scoped_to_aux_loss_layers_only` below for a
    stack where they diverge.
    """
    from src.utils.config import ModelConfig

    return ModelConfig(
        d_model=64,
        n_layers=4,
        vocab_size=256,
        attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4, "n_kv_heads": 2}}],
        mlp=[
            {
                "mlp_cls": "dense",
                "mlp_kwargs": {"intermediate_size": 128},
                "layer_idx": [0],
            },
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {
                    "intermediate_size": 128,
                    "n_routed_experts": 4,
                    "n_routed_experts_per_token": 2,
                    "aux_loss": True,
                    "aux_loss_coef": 1e-3,
                },
            },
        ],
    )


def test_moe_maxvio_labels_skip_dense_layer():
    """Dense-first stack (layer 0 dense, layers 1-3 MoE): maxvio metrics must be
    labeled by true model layer index. No `layer_0` should appear; the three MoE
    layers are layer_1/layer_2/layer_3.
    """
    from src.utils.config import TrainConfig, TrainingConfig

    model = _dense_first_moe_rest_cfg()
    cfg = TrainConfig(model=model, training=TrainingConfig(mixed_precision="bf16"))
    cfg.task = "pretrain"
    cfg.logging.log_every = 1
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    built = build_model(cfg)
    opt = torch.optim.SGD(built.parameters(), lr=1e-3)

    # --- train ---
    tracker.train_begin()
    _moe_on_step(tracker, built, opt, step=0)
    dt = tracker.log_train(step=1, model=built, optimizer=opt)
    train_keys = {k for k in dt if k.startswith("train-moe/maxvio_batch/")}
    assert "train-moe/maxvio_batch/layer_0" not in train_keys
    for i in (1, 2, 3):
        assert f"train-moe/maxvio_batch/layer_{i}" in train_keys

    # --- eval ---
    built.eval()
    tracker.eval_begin()
    logits = torch.randn(1, 4, model.vocab_size)
    labels = torch.randint(0, model.vocab_size, (1, 4))
    tracker.on_eval_step(loss=2.0, logits=logits, labels=labels, model=built)
    de = tracker.log_eval(step=1)
    for variant in ("maxvio_global", "maxvio_batch"):
        assert f"val-moe/{variant}/layer_0" not in de
        for i in (1, 2, 3):
            assert f"val-moe/{variant}/layer_{i}" in de


def test_metrics_moe_facts_with_dense_first_layer():
    from src.utils.config import TrainConfig, TrainingConfig

    model = _dense_first_moe_rest_cfg()
    cfg = TrainConfig(model=model, training=TrainingConfig(mixed_precision="bf16"))
    assert cfg.model.is_moe is True
    m = cfg.model
    moe_kwargs = [
        m.resolve_mlp(i)[1] for i in range(m.n_layers) if m.resolve_mlp(i)[0] == "moe"
    ]
    # aux floor counts only MoE layers: 3 layers * k(2) = 6
    assert sum(kw["n_routed_experts_per_token"] for kw in moe_kwargs) == 6
    assert len(moe_kwargs) == 3


def test_metrics_tracker_aux_floor_and_n_moe_layers_on_dense_first_stack():
    """MetricsCollector itself (not just ModelConfig) must count only MoE layers.

    n_layers=4 but n_moe_layers=3 (layer 0 is dense): the old buggy
    `n_layers * k` formula would give 4*2=8; the correct per-layer-sum formula
    gives 3*2=6, coef-weighted (aux_loss_coef=1e-3) to 3*2*1e-3=6e-3. Also pins
    the mixed-stack model-summary label.
    """
    from src.utils.config import TrainConfig, TrainingConfig

    model = _dense_first_moe_rest_cfg()
    cfg = TrainConfig(model=model, training=TrainingConfig(mixed_precision="bf16"))
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())

    # 3 MoE layers * k(2) * coef(1e-3), NOT n_layers(4) * k = 8
    assert abs(tracker._aux_floor - 6e-3) < 1e-9
    assert tracker._n_moe_layers == 3


def test_metrics_tracker_aux_floor_scoped_to_aux_loss_layers_only():
    """_aux_floor must sum k only over layers using aux_loss, not every MoE
    layer: the model's reported aux loss sums only aux_loss-layer contributions,
    so a floor computed over ALL MoE layers (including expert_bias ones, which
    contribute nothing to that sum) over-counts.

    4-layer stack: layer 0 MoE+aux_loss (k=2, coef=1e-3), layers 1-3
    MoE+expert_bias (k=2 each). All-MoE-layers formula (the bug): 4 layers *
    k(2) = 8. Correct: only layer 0 counts -> 1 * k(2) * coef(1e-3) = 2e-3.
    """
    from src.utils.config import ModelConfig, TrainConfig, TrainingConfig

    model = ModelConfig(
        d_model=64,
        n_layers=4,
        vocab_size=256,
        attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4, "n_kv_heads": 2}}],
        mlp=[
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {
                    "intermediate_size": 128,
                    "n_routed_experts": 4,
                    "n_routed_experts_per_token": 2,
                    "aux_loss": True,
                    "aux_loss_coef": 1e-3,
                },
                "layer_idx": [0],
            },
            {
                "mlp_cls": "moe",
                "mlp_kwargs": {
                    "intermediate_size": 128,
                    "n_routed_experts": 4,
                    "n_routed_experts_per_token": 2,
                    "expert_bias": True,
                },
            },
        ],
    )
    cfg = TrainConfig(model=model, training=TrainingConfig(mixed_precision="bf16"))
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())

    # only the 1 aux_loss layer * k(2) * coef(1e-3), NOT 4*2=8
    assert abs(tracker._aux_floor - 2e-3) < 1e-9
    assert tracker._n_moe_layers == 4  # expert-load logging still covers all MoE layers


def test_print_model_summary_dense_first_moe_rest_label(capsys):
    from src.utils.config import TrainConfig, TrainingConfig

    model = _dense_first_moe_rest_cfg()
    cfg = TrainConfig(model=model, training=TrainingConfig(mixed_precision="bf16"))
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    tracker.print_model_summary()
    out = capsys.readouterr().out
    assert "Model: gqa+dense/moe" in out


def _mha_first_gqa_rest_cfg():
    """4-layer mixed-attn stack: layer 0 mha, layers 1-3 gqa. Both use n_heads=2
    on d_model=64 (head_dim=32), so the shared-rope-head-dim check agrees."""
    from src.utils.config import ModelConfig

    return ModelConfig(
        d_model=64,
        n_layers=4,
        vocab_size=256,
        attn=[
            {"attn_cls": "mha", "attn_kwargs": {"n_heads": 2}, "layer_idx": [0]},
            {"attn_cls": "gqa", "attn_kwargs": {"n_heads": 2, "n_kv_heads": 1}},
        ],
        mlp=[{"mlp_cls": "dense", "mlp_kwargs": {}}],
    )


def test_print_model_summary_mixed_attn_label(capsys):
    """Mixed attn classes across layers render as a sorted 'cls/cls+...' label."""
    from src.utils.config import TrainConfig, TrainingConfig

    model = _mha_first_gqa_rest_cfg()
    cfg = TrainConfig(model=model, training=TrainingConfig(mixed_precision="bf16"))
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    tracker.print_model_summary()
    out = capsys.readouterr().out
    assert "Model: gqa/mha+dense" in out


# ---------------------------------------------------------------------------
# print_model_summary
# ---------------------------------------------------------------------------


def test_print_model_summary_dense(capsys):
    cfg = TrainConfig(max_seq_len=128, model=_cfg().model)
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    assert tracker.print_model_summary() is None
    out = capsys.readouterr().out
    assert "Model: mha+dense" in out and "non-embedding" in out and "device=cpu" in out


def test_print_model_summary_moe(capsys):
    cfg = TrainConfig(
        max_seq_len=128,
        model=ModelConfig(
            n_layers=2,
            d_model=64,
            vocab_size=256,
            attn=[
                {
                    "attn_cls": "gqa",
                    "attn_kwargs": {"n_heads": 2, "n_kv_heads": 1, "qk_norm": True},
                }
            ],
            mlp=[
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "n_routed_experts": 4,
                        "aux_loss": True,
                        "aux_loss_coef": 1e-3,
                        "n_routed_experts_per_token": 2,
                        "intermediate_size": 128,
                    },
                }
            ],
        ),
    )
    tracker = MetricsCollector(cfg, device="cpu", logger=FakeLogger())
    tracker.print_model_summary()
    out = capsys.readouterr().out
    assert "total params" in out and "active non-embedding" in out


# ---------------------------------------------------------------------------
# Activation monitoring
# ---------------------------------------------------------------------------
# Nothing is recorded unless the window is armed; the statistic is exact under
# gradient accumulation (sums, not averaged metrics); fp32 accumulation survives
# a bf16 forward; and installing on a compiled model must not thrash recompiles.


D_MODEL = 64


def _tracker_for_activations(log_every=2, **logging):
    """MetricsCollector wired for the activation-window tests."""
    return _tracker(_cfg(log_every=log_every, **logging))


@pytest.fixture(autouse=True)
def _disarm_recording():
    """_RECORDING is module-global: without this, one failing test leaves recording
    armed for every later test in the same xdist worker."""
    yield
    set_activation_monitoring_status(False)


def _block_cfg(n_layers=1):
    return ModelConfig(
        d_model=D_MODEL,
        n_layers=n_layers,
        vocab_size=256,
        attn=[
            {
                "attn_cls": "mha",
                "attn_kwargs": {"n_heads": 4, "attn_implementation": "sdpa"},
            }
        ],
        mlp=[{"mlp_cls": "dense", "mlp_kwargs": {"activation": "silu", "gated": True}}],
        norm_cls="rmsnorm",
        pos_emb_cls="rope",
        pos_emb_kwargs={},
    )


def _run_block(block, x=None):
    x = torch.randn(2, 16, D_MODEL) if x is None else x
    pos = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
    attn_mask, _ = make_attn_mask("causal", "sdpa", pos, x.dtype)
    return block(x, attn_mask=attn_mask)


def _rms(t):
    return t.double().pow(2).mean().sqrt().item()


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_nothing_recorded_without_scope():
    """Training must be unaffected unless a scope is open."""
    model = torch.nn.Linear(D_MODEL, 8)
    apply_activation_monitoring(model)
    model(torch.randn(4, D_MODEL))
    assert compute_activation_norm(model) == {}


def test_scope_records_linear_input():
    model = torch.nn.Linear(D_MODEL, 8)
    apply_activation_monitoring(model)
    x = torch.randn(4, D_MODEL)
    set_activation_monitoring_status(True)
    model(x)
    assert compute_activation_norm(model)["input"] == pytest.approx(_rms(x), rel=1e-5)


def test_reset_clears_accumulators():
    model = torch.nn.Linear(D_MODEL, 8)
    apply_activation_monitoring(model)
    set_activation_monitoring_status(True)
    model(torch.randn(4, D_MODEL))
    reset_activation_stats(model)
    assert compute_activation_norm(model) == {}


# ---------------------------------------------------------------------------
# Exactness under gradient accumulation
# ---------------------------------------------------------------------------


def test_accumulates_exactly_across_micro_batches():
    """The reason to accumulate sums and not averaged metrics: N micro-batches must
    give the RMS of their concatenation, not the mean of their RMSs."""
    model = torch.nn.Linear(D_MODEL, 8)
    apply_activation_monitoring(model)
    micro = [torch.randn(4, D_MODEL) * scale for scale in (0.1, 1.0, 10.0)]
    set_activation_monitoring_status(True)
    for x in micro:
        model(x)
    expected = _rms(torch.cat(micro))
    assert compute_activation_norm(model)["input"] == pytest.approx(expected, rel=1e-5)
    # and it is genuinely different from averaging per-micro-batch RMS
    assert expected != pytest.approx(sum(_rms(x) for x in micro) / len(micro), rel=1e-3)


def test_accumulates_in_fp32_from_bf16_forward():
    """A bf16 sum over millions of elements would drift; the accumulator is fp32."""
    model = torch.nn.Linear(D_MODEL, 8).to(torch.bfloat16)
    apply_activation_monitoring(model)
    x = torch.randn(256, D_MODEL, dtype=torch.bfloat16)
    set_activation_monitoring_status(True)
    model(x)
    assert model.act_stats.energy.dtype == torch.float32
    got = compute_activation_norm(model)["input"]
    assert got == pytest.approx(_rms(x), rel=1e-2)
    assert math.isfinite(got)


# ---------------------------------------------------------------------------
# Sites and naming
# ---------------------------------------------------------------------------


def test_sites_are_exactly_the_linear_inputs():
    """Every site is one Linear's GEMM operand, named by its module path. Anything
    that is not a Linear input -- the residual stream, the SwiGLU product -- is not
    reachable by a hook and is deliberately absent."""
    block = TransformerBlock(_block_cfg(), layer_idx=0)
    apply_activation_monitoring(block)
    set_activation_monitoring_status(True)
    _run_block(block)
    names = set(compute_activation_norm(block))
    assert names == {
        f"{n}.input" for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)
    }
    assert "mlp.down_proj.input" in names


def test_embedding_is_not_a_site_but_tied_lm_head_is():
    """torch.nn.Embedding consumes token ids, not an activation; the tied lm_head still
    contracts a real activation against the same weight."""
    model = torch.nn.Module()
    model.token_emb = torch.nn.Embedding(32, D_MODEL)
    model.lm_head = torch.nn.Linear(D_MODEL, 32, bias=False)
    model.lm_head.weight = model.token_emb.weight
    apply_activation_monitoring(model)
    set_activation_monitoring_status(True)
    model.lm_head(model.token_emb(torch.randint(0, 32, (2, 8))))
    assert set(compute_activation_norm(model)) == {"lm_head.input"}


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------


def test_compiled_model_records_and_does_not_thrash_recompiles(device):
    """Hooks must be installed before compile, and must not close over per-module
    data -- a per-name closure specializes per site and blows the recompile limit."""
    if device != "cuda":
        pytest.skip("compile parity checked on CUDA")
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    model = torch.nn.Sequential(
        *[torch.nn.Linear(D_MODEL, D_MODEL) for _ in range(6)]
    ).to(device)
    apply_activation_monitoring(model)
    compiled = torch.compile(model)
    x = torch.randn(4, D_MODEL, device=device)
    compiled(x)
    set_activation_monitoring_status(True)
    compiled(x)
    norms = compute_activation_norm(model)
    assert len(norms) == 6
    assert counters["stats"]["unique_graphs"] <= 3, counters["stats"]
    assert not counters["graph_break"], dict(counters["graph_break"])


# ---------------------------------------------------------------------------
# Wiring: cadence, emission, reset
# ---------------------------------------------------------------------------


def _linear_model():
    model = torch.nn.Linear(D_MODEL, 8)
    apply_activation_monitoring(model)
    return model


def test_records_only_on_the_step_whose_metrics_are_logged():
    """Same gate as snapshot_pre_step: accumulate during the step whose update is
    about to be logged, so the window is exactly one optimizer step."""
    collector, _ = _tracker_for_activations(log_every=2, log_activation_norms=True)
    model = _linear_model()
    for step in (0, 1):
        collector.train_step_begin(model, step)
        model(torch.randn(4, D_MODEL))
    # step 1 is the pre-log step ((1 + 1) % 2 == 0); step 0 must not contribute
    assert model.act_stats.count.item() == 4 * D_MODEL


def test_window_is_re_derived_every_step_so_a_leak_self_heals():
    """No context manager guards the window, so arming must be idempotent: an
    exception mid-step cannot leave recording on past the next step's arm call."""
    collector, _ = _tracker_for_activations(log_every=2, log_activation_norms=True)
    model = _linear_model()
    collector.train_step_begin(model, 1)  # armed
    model(torch.randn(4, D_MODEL))
    collector.train_step_begin(model, 2)  # (2 + 1) % 2 != 0 -> disarmed
    model(torch.randn(4, D_MODEL))
    assert model.act_stats.count.item() == 4 * D_MODEL


def test_disabled_flag_records_nothing():
    collector, _ = _tracker_for_activations(log_every=1, log_activation_norms=False)
    model = _linear_model()
    collector.train_step_begin(model, 0)
    model(torch.randn(4, D_MODEL))
    assert compute_activation_norm(model) == {}


def test_arming_resets_so_each_window_starts_clean():
    """train_step_begin owns the whole window: it resets as it arms, so log_train
    never has to mutate the model and a window can't inherit the previous one."""
    collector, _ = _tracker_for_activations(log_every=1, log_activation_norms=True)
    model = _linear_model()
    collector.train_step_begin(model, 0)
    model(torch.randn(4, D_MODEL))
    collector.train_step_begin(model, 1)  # next window
    model(torch.randn(7, D_MODEL))
    assert model.act_stats.count.item() == 7 * D_MODEL


def test_forwards_between_windows_are_discarded_not_logged():
    """Eval runs after log_train, before the next arm, with recording still on.
    Those forwards accumulate but must never reach a log: the reset at the next
    arm drops them."""
    collector, _ = _tracker_for_activations(log_every=1, log_activation_norms=True)
    model = _linear_model()
    collector.train_step_begin(model, 0)
    model(torch.randn(4, D_MODEL))
    collector.log_train(
        step=1, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1)
    )
    model(torch.randn(9, D_MODEL))  # stands in for the eval loop
    collector.train_step_begin(model, 1)
    model(torch.randn(4, D_MODEL))
    assert model.act_stats.count.item() == 4 * D_MODEL


def test_log_train_emits_norm_keys_without_mutating_the_model():
    collector, _ = _tracker_for_activations(log_every=1, log_activation_norms=True)
    model = _linear_model()
    collector.train_step_begin(model, 0)
    model(torch.randn(4, D_MODEL))
    before = model.act_stats.count.item()
    logged = collector.log_train(
        step=1, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1)
    )
    assert "activation/norm/input" in logged
    assert logged["activation/norm/input"] > 0
    assert model.act_stats.count.item() == before  # read-only


def test_log_train_omits_norm_keys_when_disabled():
    collector, _ = _tracker_for_activations(log_every=1, log_activation_norms=False)
    model = _linear_model()
    logged = collector.log_train(
        step=1, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1)
    )
    assert not [k for k in logged if k.startswith("activation/norm/")]


def test_buffers_follow_the_model_not_the_ambient_default_device():
    """Accumulators must be created on the module's own weight device. conftest sets
    torch.set_default_device('cuda') autouse, so a bare torch.zeros(()) would take
    the process default instead -- and eager quietly cross-device syncs it, leaving
    torch.compile to be the first thing that fails, on a fake tensor mismatch."""
    block = TransformerBlock(_block_cfg(), layer_idx=0).to("cpu")
    apply_activation_monitoring(block)
    assert block.mlp.down_proj.act_stats.energy.device.type == "cpu"
    assert block.attn.q_proj.act_stats.count.device.type == "cpu"
