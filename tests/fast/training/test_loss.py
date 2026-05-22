"""Tests for compute_loss with -100 ignore_index convention."""

import torch
import torch.nn.functional as F

from src.training.loss import compute_loss


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
