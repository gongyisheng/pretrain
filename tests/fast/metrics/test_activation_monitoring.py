"""Tests for activation monitoring — the accumulators, their install pass, and the
window the collector opens around them.

Nothing is recorded unless the window is armed; the statistic is exact under gradient
accumulation (sums, not averaged metrics); fp32 accumulation survives a bf16 forward;
and installing on a compiled model must not thrash recompiles.
"""

import math

import pytest
import torch

from src.layers.block import TransformerBlock
from src.metrics.activation import (
    reset_activation_stats,
    set_activation_monitoring_status,
)
from src.metrics.collector import MetricsCollector
from src.metrics.convert import apply_activation_monitoring
from src.metrics.functional import compute_activation_norm
from src.utils.config import ModelConfig, TrainConfig
from tests.fast.helpers import make_attn_mask


D_MODEL = 64


class FakeLogger:
    """Captures (step, dict) calls in place of WandbLogger."""

    def __init__(self):
        self.logs: list[tuple[int, dict]] = []

    def log(self, metrics: dict, step: int):
        self.logs.append((step, metrics))

    def log_text(self, key: str, text: str, step: int):
        pass

    def finish(self):
        pass


def _cfg(log_every=2, **logging):
    cfg = TrainConfig(
        model=ModelConfig(
            n_layers=2,
            d_model=D_MODEL,
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
    cfg.logging.log_every = log_every
    for key, value in logging.items():
        setattr(cfg.logging, key, value)
    return cfg


def _tracker_for_activations(log_every=2, **logging):
    """MetricsCollector wired for the activation-window tests."""
    cfg = _cfg(log_every=log_every, **logging)
    return MetricsCollector(cfg, device="cpu", logger=FakeLogger()), cfg


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


def _linear_model():
    """One Linear under a container, so its key is a real module path ("0")."""
    model = torch.nn.Sequential(torch.nn.Linear(D_MODEL, 8))
    apply_activation_monitoring(model)
    return model


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_nothing_recorded_without_scope():
    """Training must be unaffected unless a scope is open."""
    model = _linear_model()
    model(torch.randn(4, D_MODEL))
    assert compute_activation_norm(model) == {}


def test_scope_records_linear_input():
    model = _linear_model()
    x = torch.randn(4, D_MODEL)
    set_activation_monitoring_status(True)
    model(x)
    assert compute_activation_norm(model)["0"] == pytest.approx(_rms(x), rel=1e-5)


def test_reset_clears_accumulators():
    model = _linear_model()
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
    model = _linear_model()
    micro = [torch.randn(4, D_MODEL) * scale for scale in (0.1, 1.0, 10.0)]
    set_activation_monitoring_status(True)
    for x in micro:
        model(x)
    expected = _rms(torch.cat(micro))
    assert compute_activation_norm(model)["0"] == pytest.approx(expected, rel=1e-5)
    # and it is genuinely different from averaging per-micro-batch RMS
    assert expected != pytest.approx(sum(_rms(x) for x in micro) / len(micro), rel=1e-3)


def test_accumulates_in_fp32_from_bf16_forward():
    """A bf16 sum over millions of elements would drift; the accumulator is fp32."""
    model = torch.nn.Sequential(torch.nn.Linear(D_MODEL, 8)).to(torch.bfloat16)
    apply_activation_monitoring(model)
    x = torch.randn(256, D_MODEL, dtype=torch.bfloat16)
    set_activation_monitoring_status(True)
    model(x)
    assert model[0].act_stats.energy.dtype == torch.float32
    got = compute_activation_norm(model)["0"]
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
        n for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)
    }
    assert "mlp.down_proj" in names


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
    assert set(compute_activation_norm(model)) == {"lm_head"}


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


def test_records_only_on_the_step_whose_metrics_are_logged():
    """Same gate as snapshot_pre_step: accumulate during the step whose update is
    about to be logged, so the window is exactly one optimizer step."""
    collector, _ = _tracker_for_activations(log_every=2, log_activation_norms=True)
    model = _linear_model()
    for step in (0, 1):
        collector.train_step_begin(model, step)
        model(torch.randn(4, D_MODEL))
    # step 1 is the pre-log step ((1 + 1) % 2 == 0); step 0 must not contribute
    assert model[0].act_stats.count.item() == 4 * D_MODEL


def test_window_is_re_derived_every_step_so_a_leak_self_heals():
    """No context manager guards the window, so arming must be idempotent: an
    exception mid-step cannot leave recording on past the next step's arm call."""
    collector, _ = _tracker_for_activations(log_every=2, log_activation_norms=True)
    model = _linear_model()
    collector.train_step_begin(model, 1)  # armed
    model(torch.randn(4, D_MODEL))
    collector.train_step_begin(model, 2)  # (2 + 1) % 2 != 0 -> disarmed
    model(torch.randn(4, D_MODEL))
    assert model[0].act_stats.count.item() == 4 * D_MODEL


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
    assert model[0].act_stats.count.item() == 7 * D_MODEL


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
    assert model[0].act_stats.count.item() == 4 * D_MODEL


def test_log_train_emits_norm_keys_without_mutating_the_model():
    collector, _ = _tracker_for_activations(log_every=1, log_activation_norms=True)
    model = _linear_model()
    collector.train_step_begin(model, 0)
    model(torch.randn(4, D_MODEL))
    before = model[0].act_stats.count.item()
    logged = collector.log_train(
        step=1, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1)
    )
    assert "activation/norm/0" in logged
    assert logged["activation/norm/0"] > 0
    assert model[0].act_stats.count.item() == before  # read-only


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
