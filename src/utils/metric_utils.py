"""Pure metric computation — stateless functions, no logging, no state.

The bottom layer of the metrics stack. Everything here takes numbers/tensors
in and returns numbers/dicts out; it never touches W&B or the training loop.
`MetricsTracker` (the stateful assembler) builds on these primitives.
"""

import math
import statistics

import torch

from src.model.transformer import TransformerLM
from src.utils.config import TrainConfig

# ---------------------------------------------------------------------------
# Gradient norms
# ---------------------------------------------------------------------------


def compute_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """
    Per-parameter L2 gradient norms, keyed by parameter name.
    """
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        # torch.compile wraps model in OptimizedModule, prepending "_orig_mod."
        name = name.removeprefix("_orig_mod.")
        norms[name] = param.grad.data.norm(2.0).item()
    return norms


# ---------------------------------------------------------------------------
# MoE load balance (MaxVio, arXiv:2408.15664)
# ---------------------------------------------------------------------------


def compute_maxvio(expert_counts: torch.Tensor) -> float:
    """Maximal load violation for one MoE layer (arXiv:2408.15664):
    `(max_i load_i - mean load) / mean load`. 0 = perfectly balanced; 1.0 means
    the hottest expert carries 2x its fair share.
    """
    counts = expert_counts.float()
    mean = counts.mean()
    if mean == 0:
        return 0.0
    return ((counts.max() - mean) / mean).item()


def aggregate_maxvio(per_layer: list[float]) -> dict[str, float]:
    """Wrap per-layer MaxVio values into ``{"layer_{i}": v, "mean", "max"}``."""
    out = {f"layer_{i}": v for i, v in enumerate(per_layer)}
    out["mean"] = statistics.mean(per_layer)
    out["max"] = max(per_layer)
    return out


def compute_moe_maxvio(load_per_layer: list[torch.Tensor]) -> dict[str, float]:
    """Per-layer and aggregate MaxVio from accumulated expert load counts.

    Returns ``{"layer_{i}": v, ..., "mean": ..., "max": ...}``.
    """
    return aggregate_maxvio([compute_maxvio(c) for c in load_per_layer])


def compute_moe_global_maxvio(load_per_layer: list[torch.Tensor]) -> dict[str, float]:
    """MaxVio of the load summed over batches, from stacked ``[n_batch, E]``
    counts per layer, then aggregate.
    """
    return aggregate_maxvio([compute_maxvio(loads.sum(0)) for loads in load_per_layer])


def compute_moe_batch_maxvio(load_per_layer: list[torch.Tensor]) -> dict[str, float]:
    """Mean per-batch MaxVio from stacked ``[n_batch, E]`` counts per layer,
    then aggregate.
    """
    per_layer = [
        statistics.mean([compute_maxvio(row) for row in loads])
        for loads in load_per_layer
    ]
    return aggregate_maxvio(per_layer)


# ---------------------------------------------------------------------------
# Weight spectral metrics (SVD)
# ---------------------------------------------------------------------------


def _esd_alpha(eigs: torch.Tensor) -> float:
    """Power-law exponent α of the ESD tail (Martin & Mahoney / WeightWatcher).

    Fits p(λ) ∝ λ^(-α) to the upper tail λ ≥ xmin of the eigenvalue spectrum
    (eigs = σ² of the weight). For each candidate xmin (every eigenvalue) the
    continuous MLE α = 1 + n / Σ ln(λ_tail / xmin) is computed, and the xmin
    minimizing the KS distance between the empirical and fitted tail CDFs is
    chosen. Lower α (heavier tail, ~2–4) tracks a more correlated / better-fit
    weight; α → large signals a near-random spectrum. Returns NaN if fewer than
    four positive eigenvalues survive.
    """
    eigs = torch.sort(eigs.flatten()).values
    eigs = eigs[eigs > 0]
    n = eigs.numel()
    if n < 4:
        return float("nan")
    logs = eigs.log()
    # suffix[i] = Σ_{j>=i} log λ_j  →  Σ_{j>=i} ln(λ_j / λ_i) = suffix[i] - nt·logs[i]
    suffix = torch.flip(torch.cumsum(torch.flip(logs, [0]), 0), [0])
    idx = torch.arange(n, device=eigs.device)
    nt = (n - idx).to(logs.dtype)  # tail size for xmin = eigs[i]
    alpha = 1.0 + nt / (suffix - nt * logs).clamp_min(1e-12)
    # KS distance per candidate xmin: max over the tail of |emp_cdf - fit_cdf|.
    log_ratio = logs.unsqueeze(0) - logs.unsqueeze(1)  # [i,j] = ln(λ_j / λ_i)
    fit_cdf = 1.0 - torch.exp(-(alpha.unsqueeze(1) - 1.0) * log_ratio)
    rank = (idx.unsqueeze(0) - idx.unsqueeze(1) + 1).to(logs.dtype)  # [i,j]=j-i+1
    emp_cdf = rank / nt.unsqueeze(1)
    tail = idx.unsqueeze(0) >= idx.unsqueeze(1)  # j >= i
    ks = torch.where(tail, (emp_cdf - fit_cdf).abs(), logs.new_zeros(())).amax(1)
    ks = torch.where(nt >= 2, ks, logs.new_full((), float("inf")))
    return alpha[ks.argmin()].item()


def _svd_metrics(weight: torch.Tensor) -> dict[str, float]:
    """srank / pr / esd_alpha of a weight's σ² spectrum.

    srank = stable rank (‖W‖_F² / σ_max²), top-heavy — a rank-1 collapse canary.
    pr = participation ratio ((Σσ²)² / Σσ⁴ = 1/Σpᵢ²), the bulk effective
    dimension; squaring suppresses the noise floor for a cleaner collapse trend.
    esd_alpha = power-law exponent of the eigenvalue tail (see `_esd_alpha`).

    srank for monitoring rank-1 collapse
    pr for monitoring graded collapse
    esd_alpha for monitoring heavy-tailed self-regularization

    Accepts a 2D matrix or a stacked 3D tensor `(E, out, in)` (MoE experts);
    for 3D the SVD is batched over the leading dim and the per-expert metrics
    are averaged.
    """
    s = torch.linalg.svdvals(weight.float())  # (..., k)
    energy = s.pow(2)
    total = energy.sum(-1)
    if (total == 0).any():
        return {"srank": 0.0, "pr": 0.0, "esd_alpha": 0.0}
    srank = total / energy[..., 0]
    pr = total.pow(2) / energy.pow(2).sum(-1)
    alphas = [_esd_alpha(e) for e in energy.reshape(-1, energy.shape[-1])]
    alphas = [a for a in alphas if not math.isnan(a)]
    esd_alpha = statistics.mean(alphas) if alphas else float("nan")
    return {
        "srank": srank.mean().item(),
        "pr": pr.mean().item(),
        "esd_alpha": esd_alpha,
    }


def compute_layer_svd_metrics(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Per-weight spectral metrics, keyed by parameter name.

    Covers every floating-point 2D parameter and stacked 3D MoE expert tensors
    `(E, out, in)` (averaged per-expert), except rope buffers and embeddings
    (names containing "emb"). The caller applies the logging namespace (e.g. an
    "optim/" prefix). The "_orig_mod." prefix added by torch.compile is
    stripped. SVD on every weight is costly, so this is gated behind
    config.logging.log_optimizer_svd_metrics and only runs on log-cadence steps.
    """
    metrics: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        if param.ndim not in (2, 3) or not param.is_floating_point():
            continue
        name = name.removeprefix("_orig_mod.")
        if name.startswith("rope.") or "emb" in name:
            continue
        metrics[name] = _svd_metrics(param.detach())
    return metrics


# ---------------------------------------------------------------------------
# Optimizer step diagnostics (||Δθ||, ||m||, ||v||)
# ---------------------------------------------------------------------------


def snapshot_params(model: torch.nn.Module) -> list[torch.Tensor]:
    """
    Detached clone of every trainable parameter, ordered by model.parameters().

    Use with compute_param_step_norm to measure ||θ_after - θ_before||.
    """
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def compute_param_step_norm(
    model: torch.nn.Module, snapshot: list[torch.Tensor]
) -> float:
    """
    L2 norm of the parameter delta vs. an earlier snapshot.
    """
    total_sq: torch.Tensor | None = None
    params = (p for p in model.parameters() if p.requires_grad)
    for p, p_before in zip(params, snapshot, strict=True):
        sq = (p.detach() - p_before).pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    return total_sq.sqrt().item() if total_sq is not None else 0.0


def _aggregate_state_norm(
    optimizer: torch.optim.Optimizer, *, key: str
) -> float | None:
    """
    L2 norm across all `state[p][key]` buffers, or None if no buffer exists.
    """
    total_sq: torch.Tensor | None = None
    for state in optimizer.state.values():
        buf = state.get(key)
        if buf is None:
            continue
        sq = buf.pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    return total_sq.sqrt().item() if total_sq is not None else None


def compute_momentum_norm(optimizer: torch.optim.Optimizer) -> float | None:
    """
    L2 norm across optimizer first-moment buffers.
    """
    return _aggregate_state_norm(optimizer, key="exp_avg")


def compute_variance_norm(optimizer: torch.optim.Optimizer) -> float | None:
    """
    L2 norm across optimizer second-moment buffers.
    """
    return _aggregate_state_norm(optimizer, key="exp_avg_sq")


# ---------------------------------------------------------------------------
# Generic reducers
# ---------------------------------------------------------------------------


def compute_statistics(values: list[float]) -> dict[str, float]:
    """
    {mean, median, max, min} over a non-empty list of scalars.
    """
    if not values:
        return {}
    return {
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "max": max(values),
        "min": min(values),
    }


# ---------------------------------------------------------------------------
# Eval primitives
# ---------------------------------------------------------------------------


def count_correct(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    exclude_id: int | None = None,
) -> tuple[int, int]:
    """
    Count positions where argmax(logits) == labels.
    """
    preds = logits.argmax(dim=-1)
    mask = labels != ignore_index
    if exclude_id is not None:
        mask = mask & (labels != exclude_id)
    correct = (preds[mask] == labels[mask]).sum().item()
    total = int(mask.sum().item())
    return int(correct), total


def compute_decoded_byte_len(tokenizer, token_ids: list[int]) -> int:
    """
    UTF-8 byte length of `token_ids` decoded back to text.
    """
    return len(tokenizer.decode(token_ids, skip_special_tokens=True).encode("utf-8"))


def compute_bytes_per_token(tokenizer, texts: list[str]) -> float:
    """
    Bytes/token over `texts` using `tokenizer` (special tokens excluded).
    """
    n_bytes = 0
    n_tokens = 0
    for t in texts:
        n_bytes += len(t.encode("utf-8"))
        n_tokens += len(tokenizer.encode(t, add_special_tokens=False).ids)
    if n_tokens == 0:
        raise ValueError("no tokens produced; corpus may be empty")
    return n_bytes / n_tokens


def compute_perplexity(loss: float, cap: float = 1e6) -> float:
    """exp(loss), capped to avoid Inf on early high-loss steps.

    Uses torch.exp so overflow saturates to the cap rather than raising.
    """
    return min(float(torch.exp(torch.tensor(loss))), cap)


def compute_bits_per_byte(loss: float, tokens_per_byte: float) -> float:
    """Convert a nats/token loss to bits/byte: loss * tokens_per_byte / ln(2)."""
    return loss * tokens_per_byte / math.log(2)


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------


def count_parameters(config: TrainConfig) -> dict[str, int]:
    """
    Total, non-embedding, and active-non-embedding parameter counts.
    """
    return TransformerLM.compute_parameters(config.model, config.max_seq_len)


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def compute_flops_per_token(config: TrainConfig) -> int:
    """
    Total training FLOPs per token
    """
    fwd_total = TransformerLM.compute_flops(config.model, config.max_seq_len)
    backward_mult = 4 if config.training.activation_checkpointing else 3
    return fwd_total * backward_mult


# ---------------------------------------------------------------------------
# GPU peak FLOPS estimation (for MFU)
# ---------------------------------------------------------------------------


def estimate_gpu_peak_flops(device: str) -> float | None:
    """Estimate GPU peak bf16/fp16 FLOPS for MFU calculation.

    Returns None on non-CUDA devices or unrecognized GPUs (caller skips MFU).
    """
    if device != "cuda":
        return None
    name = torch.cuda.get_device_properties(0).name.lower()
    # bf16/fp16 tensor-core peak TFLOPS for common GPUs
    gpu_tflops = [
        ("h100-sxm", 990),
        ("h100", 756),
        ("a100-sxm", 312),
        ("a100-pcie", 250),
        ("a100", 312),
        ("l40s", 362),
        ("l40", 181),
        ("rtx pro 6000 blackwell", 503.8),
        ("rtx 6000 ada", 364.2),
        ("5090", 104.8),
        ("5080", 56.28),
        ("5060 ti", 23.7),
        ("4090", 165),
        ("4080", 97),
        ("3090", 71),
        ("3080", 47),
    ]
    for key, tflops in gpu_tflops:
        if key in name:
            return tflops * 1e12 * torch.cuda.device_count()
    return None
