import torch
import torch.nn.functional as F


def shift_inputs(tokens: torch.Tensor) -> torch.Tensor:
    """Drop the last token along sequence dim — the model input for next-token prediction.

    Args:
        tokens: shape (B, S+1)
    Returns:
        x: shape (B, S)
    """
    return tokens[:, :-1]


def compute_loss(
    logits: torch.Tensor, labels: torch.Tensor, loss_fn=None
) -> torch.Tensor:
    """Compute next-token cross-entropy loss with -100 ignore-index convention.

    Args:
        logits: shape (B, S, V)
        labels: shape (B, S); positions with label == -100 are ignored.
        loss_fn: optional callable(flat_logits, flat_labels, ignore_index=-100) → loss;
            defaults to F.cross_entropy.

    Returns:
        Scalar loss tensor. NaN if every position is ignored.
    """
    if loss_fn is None:
        loss_fn = F.cross_entropy
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    return loss_fn(flat_logits, flat_labels, ignore_index=-100)
