import torch
import torch.nn.functional as F


@torch.compile
def _cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


@torch.compile
def _cross_entropy_fp64(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    # Upcast logits to fp64 for softmax+CE so the correct-class gradient does
    # not absorb to zero in high-confidence regimes — eliminates slingshot
    # spikes (Liu et al. 2025, arXiv:2605.06152).
    return F.cross_entropy(
        logits.double(),
        targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


@torch.compile
def _mse_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,  # inert for MSE; accepted for uniform call signature
) -> torch.Tensor:
    # MSE between softmax(logits) and one-hot(targets), averaged over classes
    # per sample, then averaged over non-ignored samples. Returns NaN when all
    # samples are ignored (matches F.cross_entropy with ignore_index).
    # F.mse_loss has no ignore_index, so masking is applied here.
    valid = (targets != ignore_index).to(logits.dtype)
    safe_targets = targets.clamp(min=0)
    one_hot = F.one_hot(safe_targets, num_classes=logits.size(-1)).to(logits.dtype)
    probs = F.softmax(logits, dim=-1)
    per_sample = F.mse_loss(probs, one_hot, reduction="none").mean(dim=-1)
    return (per_sample * valid).sum() / valid.sum()


@torch.compile
def _mse_loss_fp64(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,  # inert for MSE; accepted for uniform call signature
) -> torch.Tensor:
    # fp64 variant: softmax in fp64 so p_correct doesn't absorb to exactly 1.0
    # in high-confidence regimes (same motivation as _cross_entropy_fp64).
    logits = logits.double()
    valid = (targets != ignore_index).to(logits.dtype)
    safe_targets = targets.clamp(min=0)
    one_hot = F.one_hot(safe_targets, num_classes=logits.size(-1)).to(logits.dtype)
    probs = F.softmax(logits, dim=-1)
    per_sample = F.mse_loss(probs, one_hot, reduction="none").mean(dim=-1)
    return (per_sample * valid).sum() / valid.sum()


LOSS_REGISTRY = {
    "cross_entropy": _cross_entropy,
    "cross_entropy_fp64": _cross_entropy_fp64,
    "mse": _mse_loss,
    "mse_fp64": _mse_loss_fp64,
}


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: str = "cross_entropy",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute next-token loss with -100 ignore-index convention.

    Args:
        logits: shape (B, S, V)
        labels: shape (B, S); positions with label == -100 are ignored.
        loss_fn: key into LOSS_REGISTRY (e.g. "cross_entropy", "cross_entropy_fp64",
            "mse", "mse_fp64").
        label_smoothing: ε for label-smoothed CE; floors loss at high confidence.
            Inert for MSE variants (accepted for uniform call signature).

    Returns:
        Scalar loss tensor. NaN if every position is ignored.
    """
    if loss_fn not in LOSS_REGISTRY:
        raise ValueError(
            f"unknown loss_fn {loss_fn!r}; expected one of {sorted(LOSS_REGISTRY)}"
        )
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    return LOSS_REGISTRY[loss_fn](
        flat_logits, flat_labels, ignore_index=-100, label_smoothing=label_smoothing
    )
