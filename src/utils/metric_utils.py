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


def _esd_alpha_batched(energy: torch.Tensor) -> torch.Tensor:
    """Power-law ESD tail exponent α (WeightWatcher) per (M, n) σ² spectrum row; NaN if <4 positive."""
    M, n = energy.shape
    eigs, _ = torch.sort(energy, dim=-1)  # ascending → positives are the top columns
    pos = eigs > 0
    nvalid = pos.sum(-1)  # (M,) positive count per row
    tiny = torch.finfo(eigs.dtype).tiny
    logs = torch.where(pos, eigs.clamp_min(tiny).log(), eigs.new_zeros(()))
    idx = torch.arange(n, device=eigs.device)
    start = n - nvalid  # (M,) first positive column per row
    is_pos_i = idx.unsqueeze(0) >= start.unsqueeze(1)  # (M, n) column i is positive
    nt = (n - idx).to(logs.dtype).expand(M, n)  # tail size for xmin = eigs[:, i]
    # suffix[i] = Σ_{j>=i} logs[j]  (logs are 0 for non-positive, excluded columns)
    suffix = torch.flip(torch.cumsum(torch.flip(logs, [-1]), -1), [-1])
    alpha = 1.0 + nt / (suffix - nt * logs).clamp_min(1e-12)  # (M, n)
    # KS distance per candidate xmin = eigs[:, i], over its tail j >= i.
    log_ratio = logs.unsqueeze(1) - logs.unsqueeze(
        2
    )  # (M, i, j) = logs[:,j] - logs[:,i]
    fit_cdf = 1.0 - torch.exp(-(alpha.unsqueeze(2) - 1.0) * log_ratio)
    rank = (idx.view(1, 1, n) - idx.view(1, n, 1) + 1).to(logs.dtype)  # [i,j] = j-i+1
    emp_cdf = rank / nt.unsqueeze(2)
    tail = (idx.view(1, n, 1) <= idx.view(1, 1, n)) & is_pos_i.unsqueeze(
        2
    )  # j>=i, i positive
    ks = torch.where(tail, (emp_cdf - fit_cdf).abs(), eigs.new_zeros(())).amax(
        -1
    )  # (M, n)
    ks = torch.where((nt >= 2) & is_pos_i, ks, eigs.new_full((), float("inf")))
    best = ks.argmin(-1)  # (M,)
    out = alpha.gather(1, best.unsqueeze(1)).squeeze(1)
    return torch.where(nvalid >= 4, out, eigs.new_full((), float("nan")))


def _gram_energy(weight: torch.Tensor) -> torch.Tensor:
    """Descending σ² spectrum via eigvalsh of the smaller Gram matrix (= svdvals(W)², no full SVD)."""
    w = weight.float()
    m, k = w.shape[-2], w.shape[-1]
    gram = w @ w.transpose(-2, -1) if m <= k else w.transpose(-2, -1) @ w
    eig = torch.linalg.eigvalsh(gram)  # ascending, (..., min(m, k))
    return eig.clamp_min(0).flip(-1)  # descending σ²


def _svd_metrics(weight: torch.Tensor) -> dict[str, float]:
    """srank / pr / esd_alpha of a weight's σ² spectrum (averaged over experts if 3D)."""
    energy = _gram_energy(weight)  # (..., k) descending σ²
    total = energy.sum(-1)
    if (total == 0).any():
        return {"srank": 0.0, "pr": 0.0, "esd_alpha": 0.0}
    srank = total / energy[..., 0]  # stable rank ‖W‖_F²/σ_max² (rank-1 collapse canary)
    pr = total.pow(2) / energy.pow(2).sum(-1)  # participation ratio (Σσ²)²/Σσ⁴
    alphas = _esd_alpha_batched(energy.reshape(-1, energy.shape[-1]))
    alphas = alphas[~torch.isnan(alphas)]
    esd_alpha = alphas.mean().item() if alphas.numel() else float("nan")
    return {
        "srank": srank.mean().item(),
        "pr": pr.mean().item(),
        "esd_alpha": esd_alpha,
    }


def compute_layer_svd_metrics(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Spectral metrics (srank/pr/esd_alpha) per 2D/3D weight, keyed by param name (rope/embeddings skipped)."""
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
    return fwd_total * 3  # fwd + bwd (2x); no activation recomputation


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
