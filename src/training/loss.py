import torch
import torch.nn.functional as F


def next_token_targets(tokens: torch.Tensor):
    """Split a packed token sequence into model inputs and next-token targets.

    Args:
        tokens: shape (B, S+1)

    Returns:
        x: shape (B, S) — input tokens
        y: shape (B, S) — target tokens (next-token labels)
    """
    return tokens[:, :-1], tokens[:, 1:]


def compute_loss(logits, y, loss_mask=None, loss_fn=None):
    """Compute next-token prediction loss.

    Args:
        logits: shape (B, S, V)
        y: shape (B, S) — target token IDs
        loss_mask: optional bool tensor shape (B, S); when provided only True
            positions contribute to the loss (used in packing=False mode to
            exclude padding tokens)
        loss_fn: callable(logits, targets, reduction=...) → loss tensor;
            defaults to F.cross_entropy. Must support a ``reduction`` kwarg
            ('mean' or 'none') when loss_mask is provided.

    Returns:
        scalar loss tensor
    """
    if loss_fn is None:
        loss_fn = F.cross_entropy

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_y = y.reshape(-1)

    if loss_mask is None:
        return loss_fn(flat_logits, flat_y)

    per_token_loss = loss_fn(flat_logits, flat_y, reduction='none')
    mask = loss_mask.reshape(-1).float()
    return (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
