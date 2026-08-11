"""Tests for MetricsCollector — the stateful assembly layer.

Pure metric math is tested in tests/fast/metrics/test_functional.py. Here we
exercise the tracker's state machine: per-step accumulation, the log-cadence
gate, step-norm gating, and eval accumulation. A FakeLogger captures dispatched
dicts so tests assert on what would be logged without touching W&B.
"""

import math

import pytest
import torch

from src.metrics.collector import MetricsCollector
from src.metrics.convert import apply_activation_monitoring
from src.metrics.functional import compute_activation_norm
from src.metrics.quant import QuantizationStats
from src.model import build_model
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


def test_log_train_reads_quant_metrics_off_the_model():
    """Quant metrics come from accumulators living on the model, not from a log_train
    argument: the fold happens inside the training step itself."""
    tracker, _ = _tracker(_cfg(log_every=1, log_quant_metrics=True))
    model, opt, scaler = _linear_setup()
    site = QuantizationStats("fwd.act/layer_0", 1, torch.randn(1).device)
    site.src_sq += 100.0
    site.err_sq += 1.0
    site.numel += 32.0
    model.add_module("_quant_stats", torch.nn.ModuleList([site]))

    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert d["train-quant/sqnr/fwd.act/layer_0"] == pytest.approx(
        20.0
    )  # 20*log10(10/1)
    assert tracker.logger.logs[-1][1][
        "train-quant/sqnr/fwd.act/layer_0"
    ] == pytest.approx(20.0)


def test_log_train_without_quant_metrics_has_no_train_quant_keys():
    tracker, _ = _tracker(_cfg(log_every=1))
    model, opt, scaler = _linear_setup()
    tracker.train_begin()
    _do_step(tracker, model, opt, scaler, step=0)
    d = tracker.log_train(step=1, model=model, optimizer=opt)
    assert not any(k.startswith("train-quant/") for k in d)


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
# val-side activation / quantization windows
# ---------------------------------------------------------------------------


def _act_model():
    """One Linear under a container, so its metric key is a real path ("0")."""
    model = torch.nn.Sequential(torch.nn.Linear(8, 4))
    apply_activation_monitoring(model)
    return model


def _one_eval_step(tracker):
    """The minimum on_eval_step needs so log_eval has a batch to divide by."""
    tracker.on_eval_step(
        loss=2.0, logits=torch.randn(1, 2, 256), labels=torch.tensor([[1, 2]])
    )


def test_eval_begin_resets_and_arms_the_activation_window():
    tracker, _ = _tracker(_cfg(log_activation_norms=True))
    model = _act_model()
    model[0].act_stats.energy += 5.0
    model[0].act_stats.count += 2.0

    tracker.eval_begin(model)
    assert model[0].act_stats.count.item() == 0.0  # reset

    model(torch.randn(4, 8))
    assert model[0].act_stats.count.item() == 4 * 8  # armed


def test_eval_begin_leaves_the_window_shut_when_disabled():
    tracker, _ = _tracker(_cfg(log_activation_norms=False))
    model = _act_model()
    tracker.eval_begin(model)
    model(torch.randn(4, 8))
    assert model[0].act_stats.count.item() == 0.0


def test_eval_end_disarms_the_window():
    tracker, _ = _tracker(_cfg(log_activation_norms=True))
    model = _act_model()
    tracker.eval_begin(model)
    model(torch.randn(4, 8))
    tracker.eval_end()
    before = model[0].act_stats.count.item()
    model(torch.randn(7, 8))
    assert model[0].act_stats.count.item() == before


def test_forwards_after_eval_end_do_not_join_the_val_window():
    """Mirrors _evaluate's order: on SFT the train-accuracy pass runs between the val
    loop and log_eval, so the window has to be shut before it, not at the read."""
    tracker, _ = _tracker(_cfg(task="sft", log_activation_norms=True))
    model = _act_model()
    tracker.eval_begin(model)
    model(torch.randn(4, 8))  # the val loop
    tracker.eval_end()
    model(torch.randn(16, 8))  # the train-accuracy pass
    assert model[0].act_stats.count.item() == 4 * 8


def test_log_eval_emits_val_act_keys():
    tracker, _ = _tracker(_cfg(log_activation_norms=True))
    model = _act_model()
    tracker.eval_begin(model)
    model(torch.randn(4, 8))
    tracker.eval_end()
    _one_eval_step(tracker)
    d = tracker.log_eval(step=1, model=model)
    assert d["val-act/norm/0"] > 0
    assert tracker.logger.logs[-1][1]["val-act/norm/0"] > 0


def test_log_eval_omits_val_act_keys_when_disabled():
    tracker, _ = _tracker(_cfg(log_activation_norms=False))
    model = _act_model()
    tracker.eval_begin(model)
    model(torch.randn(4, 8))
    tracker.eval_end()
    _one_eval_step(tracker)
    d = tracker.log_eval(step=1, model=model)
    assert not [k for k in d if k.startswith("val-act/")]


def test_log_eval_emits_val_quant_keys():
    """Quant accumulators are filled after eval_begin, which resets them."""
    tracker, _ = _tracker(_cfg(log_quant_metrics=True))
    model = torch.nn.Linear(8, 4)
    site = QuantizationStats("fwd.act/layer_0", 1, torch.randn(1).device)
    model.add_module("_quant_stats", torch.nn.ModuleList([site]))

    tracker.eval_begin(model)
    site.src_sq += 100.0
    site.err_sq += 1.0
    site.numel += 32.0
    tracker.eval_end()
    _one_eval_step(tracker)
    d = tracker.log_eval(step=1, model=model)
    assert d["val-quant/sqnr/fwd.act/layer_0"] == pytest.approx(20.0)


def test_log_eval_omits_val_quant_keys_when_disabled():
    tracker, _ = _tracker(_cfg(log_quant_metrics=False))
    model = torch.nn.Linear(8, 4)
    site = QuantizationStats("fwd.act/layer_0", 1, torch.randn(1).device)
    site.src_sq += 100.0
    site.err_sq += 1.0
    site.numel += 32.0
    model.add_module("_quant_stats", torch.nn.ModuleList([site]))

    tracker.eval_begin(model)
    tracker.eval_end()
    _one_eval_step(tracker)
    d = tracker.log_eval(step=1, model=model)
    assert not [k for k in d if k.startswith("val-quant/")]


def test_log_eval_without_a_model_emits_no_window_keys():
    """The eight legacy call sites pass no model; nothing to read, nothing to key."""
    tracker, _ = _tracker(_cfg(log_activation_norms=True, log_quant_metrics=True))
    tracker.eval_begin()
    _one_eval_step(tracker)
    d = tracker.log_eval(step=1)
    assert not [k for k in d if k.startswith(("val-act/", "val-quant/"))]


# ---------------------------------------------------------------------------
# train-side activation window: train_step_begin arms it, log_train reads it
# ---------------------------------------------------------------------------


def test_records_only_on_the_step_whose_metrics_are_logged():
    """Same gate as snapshot_pre_step: accumulate during the step whose update is
    about to be logged, so the window is exactly one optimizer step."""
    collector, _ = _tracker(_cfg(log_every=2, log_activation_norms=True))
    model = _act_model()
    for step in (0, 1):
        collector.train_step_begin(model, step)
        model(torch.randn(4, 8))
    # step 1 is the pre-log step ((1 + 1) % 2 == 0); step 0 must not contribute
    assert model[0].act_stats.count.item() == 4 * 8


def test_window_is_re_derived_every_step_so_a_leak_self_heals():
    """No context manager guards the window, so arming must be idempotent: an
    exception mid-step cannot leave recording on past the next step's arm call."""
    collector, _ = _tracker(_cfg(log_every=2, log_activation_norms=True))
    model = _act_model()
    collector.train_step_begin(model, 1)  # armed
    model(torch.randn(4, 8))
    collector.train_step_begin(model, 2)  # (2 + 1) % 2 != 0 -> disarmed
    model(torch.randn(4, 8))
    assert model[0].act_stats.count.item() == 4 * 8


def test_disabled_flag_records_nothing():
    collector, _ = _tracker(_cfg(log_every=1, log_activation_norms=False))
    model = _act_model()
    collector.train_step_begin(model, 0)
    model(torch.randn(4, 8))
    assert compute_activation_norm(model) == {}


def test_arming_resets_so_each_window_starts_clean():
    """train_step_begin owns the whole window: it resets as it arms, so log_train
    never has to mutate the model and a window can't inherit the previous one."""
    collector, _ = _tracker(_cfg(log_every=1, log_activation_norms=True))
    model = _act_model()
    collector.train_step_begin(model, 0)
    model(torch.randn(4, 8))
    collector.train_step_begin(model, 1)  # next window
    model(torch.randn(7, 8))
    assert model[0].act_stats.count.item() == 7 * 8


def test_forwards_between_windows_are_discarded_not_logged():
    """Eval runs after log_train, before the next arm, with recording still on.
    Those forwards accumulate but must never reach a log: the reset at the next
    arm drops them."""
    collector, _ = _tracker(_cfg(log_every=1, log_activation_norms=True))
    model = _act_model()
    collector.train_step_begin(model, 0)
    model(torch.randn(4, 8))
    collector.log_train(
        step=1, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1)
    )
    model(torch.randn(9, 8))  # stands in for the eval loop
    collector.train_step_begin(model, 1)
    model(torch.randn(4, 8))
    assert model[0].act_stats.count.item() == 4 * 8


def test_log_train_emits_norm_keys_without_mutating_the_model():
    collector, _ = _tracker(_cfg(log_every=1, log_activation_norms=True))
    model = _act_model()
    collector.train_step_begin(model, 0)
    model(torch.randn(4, 8))
    before = model[0].act_stats.count.item()
    logged = collector.log_train(
        step=1, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1)
    )
    assert "train-act/norm/0" in logged
    assert logged["train-act/norm/0"] > 0
    assert model[0].act_stats.count.item() == before  # read-only


def test_log_train_omits_norm_keys_when_disabled():
    collector, _ = _tracker(_cfg(log_every=1, log_activation_norms=False))
    model = _act_model()
    logged = collector.log_train(
        step=1, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1)
    )
    assert not [k for k in logged if k.startswith("train-act/norm/")]
