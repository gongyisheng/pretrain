"""Tests for src.metrics.activation — the accumulator, the arming flag, and the
forward pre-hook.

Nothing is recorded unless the window is armed; the statistic is exact under gradient
accumulation (sums, not averaged metrics); and fp32 accumulation survives a bf16
forward.
"""

import math

import pytest
import torch

from src.metrics.activation import (
    reset_activation_stats,
    set_activation_monitoring_status,
)
from src.metrics.convert import apply_activation_monitoring
from src.metrics.functional import compute_activation_norm


D_MODEL = 64


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
