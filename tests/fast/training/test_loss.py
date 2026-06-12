"""Tests for compute_loss with -100 ignore_index convention."""

import torch
import torch.nn.functional as F

from src.training.loss import (
    LOSS_REGISTRY,
    _cross_entropy,
    _cross_entropy_fp64,
    _mse_loss,
    _mse_loss_fp64,
    compute_loss,
)
from src.utils.config import ModelTrainingConfig


def test_compute_loss_no_minus_100_matches_plain_ce():
    """With no -100s, compute_loss == F.cross_entropy."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 10)  # (B, S, V)
    labels = torch.randint(0, 10, (2, 4))
    loss = compute_loss(logits, labels)
    expected = F.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1))
    assert torch.allclose(loss, expected)


def test_compute_loss_ignores_minus_100():
    """Positions with label == -100 don't contribute to the loss."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 10)
    labels = torch.tensor([[-100, -100, -100, 3]])
    loss = compute_loss(logits, labels)
    # Should equal the CE on position 3 alone.
    expected = F.cross_entropy(logits[0, 3].unsqueeze(0), labels[0, 3].unsqueeze(0))
    assert torch.allclose(loss, expected)


def test_compute_loss_all_minus_100_returns_nan_safe():
    """All -100 → cross_entropy with ignore_index returns nan; consumers must avoid this."""
    logits = torch.randn(1, 2, 10)
    labels = torch.tensor([[-100, -100]])
    loss = compute_loss(logits, labels)
    # PyTorch's cross_entropy with full ignore returns nan; documented behavior.
    assert torch.isnan(loss).item()


def test_loss_fn_default_is_fp32_ce():
    assert ModelTrainingConfig().loss_fn == "cross_entropy"


def test_loss_registry_contains_all_variants():
    assert {"cross_entropy", "cross_entropy_fp64", "mse", "mse_fp64"} <= set(
        LOSS_REGISTRY
    )


def test_fp64_ce_avoids_softmax_absorption():
    """
    On near-saturated logits fp32 CE rounds to zero while fp64 retains it.
    ref: https://arxiv.org/pdf/2605.06152
    """
    logits = torch.zeros(1, 16, dtype=torch.float32)
    logits[0, 3] = 30.0
    targets = torch.tensor([3])
    assert _cross_entropy(logits, targets).item() == 0.0
    assert _cross_entropy_fp64(logits, targets).item() > 0.0


def test_mse_loss_matches_F_mse_on_softmax_probs():
    """_mse_loss equals F.mse_loss(softmax(logits), one_hot(targets)) when no positions are ignored."""
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    loss = _mse_loss(logits, targets)
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(targets, num_classes=10).to(probs.dtype)
    expected = F.mse_loss(probs, one_hot)
    assert torch.allclose(loss, expected)


def test_mse_loss_ignores_minus_100():
    """Positions with target == -100 don't contribute to the MSE mean."""
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    targets = torch.tensor([-100, -100, -100, 3])
    loss = _mse_loss(logits, targets)
    expected = _mse_loss(logits[3:4], targets[3:4])
    assert torch.allclose(loss, expected)


def test_mse_loss_all_minus_100_returns_nan():
    """All -100 → 0/0 → nan, matching cross-entropy behavior."""
    logits = torch.randn(1, 2, 10)
    labels = torch.tensor([[-100, -100]])
    loss = compute_loss(logits, labels, loss_fn="mse")
    assert torch.isnan(loss).item()


def test_mse_fp64_retains_precision_at_saturated_logits():
    """fp32 softmax absorbs p_correct to exactly 1, dropping the (1-p)^2 term;
    fp64 preserves it, so fp64 MSE > fp32 MSE at high confidence."""
    logits = torch.zeros(1, 16, dtype=torch.float32)
    logits[0, 3] = 30.0
    targets = torch.tensor([3])
    loss_fp32 = _mse_loss(logits, targets).item()
    loss_fp64 = _mse_loss_fp64(logits, targets).item()
    assert loss_fp64 > loss_fp32


# ==================== label smoothing ====================


def test_training_config_default_label_smoothing_is_zero():
    assert ModelTrainingConfig().label_smoothing == 0.0


def test_compute_loss_label_smoothing_matches_F_cross_entropy():
    """compute_loss(label_smoothing=ε) == F.cross_entropy(label_smoothing=ε)."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 10)
    labels = torch.randint(0, 10, (2, 4))
    loss = compute_loss(logits, labels, label_smoothing=0.1)
    expected = F.cross_entropy(
        logits.reshape(-1, 10), labels.reshape(-1), label_smoothing=0.1
    )
    assert torch.allclose(loss, expected)


def test_label_smoothing_floors_ce_at_saturated_logits():
    """With ε > 0, CE has a non-zero floor even when p_correct → 1.

    Without smoothing, a margin-30 logit gives CE ≈ 0. With ε=0.05 and V=16,
    the floor is approximately ε · log(V) ≈ 0.139 nats (the lower bound is
    the entropy of the smoothed target distribution shifted by KL terms; the
    exact value is bounded below by ε · log(V) for large margins).
    """
    logits = torch.zeros(1, 16, dtype=torch.float64)
    logits[0, 3] = 30.0
    targets = torch.tensor([3])
    loss_no_smooth = _cross_entropy_fp64(logits, targets, label_smoothing=0.0).item()
    loss_smooth = _cross_entropy_fp64(logits, targets, label_smoothing=0.05).item()
    assert loss_no_smooth < 1e-10
    assert loss_smooth > 0.1  # floor is well above zero


def test_compute_loss_label_smoothing_default_matches_no_smoothing():
    """Default label_smoothing=0.0 must reproduce the unsmoothed loss exactly."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 10)
    labels = torch.randint(0, 10, (2, 4))
    loss_default = compute_loss(logits, labels)
    loss_explicit = compute_loss(logits, labels, label_smoothing=0.0)
    assert torch.allclose(loss_default, loss_explicit)


def test_compute_loss_label_smoothing_ignored_for_mse():
    """MSE has no label-smoothing concept; the kwarg is accepted but inert."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 10)
    labels = torch.randint(0, 10, (2, 4))
    loss_with = compute_loss(logits, labels, loss_fn="mse", label_smoothing=0.1)
    loss_without = compute_loss(logits, labels, loss_fn="mse")
    assert torch.allclose(loss_with, loss_without)


def test_yaml_loads_training_label_smoothing(tmp_path):
    from src.utils.config import load_config

    yaml_content = """
max_seq_len: 128
training:
  label_smoothing: 0.05
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.training.label_smoothing == 0.05
