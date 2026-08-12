"""Pure metric computation — stateless functions, no logging, no state.

The bottom layer of the metrics stack. Everything here takes numbers/tensors
in and returns numbers/dicts out; it never touches W&B or the training loop.
`MetricsCollector` (the stateful assembler) builds on these primitives.
"""

import math
import statistics

import torch

from src.metrics.activation import ActivationStats
from src.metrics.quant import QuantizationStats
from src.model.transformer import TransformerLM
from src.quant.constants import EPS
from src.utils.config import TrainConfig

# ---------------------------------------------------------------------------
# Tensor norms
# ---------------------------------------------------------------------------


def compute_grad_norms(model: torch.nn.Module) -> dict[str, float]:
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


def compute_activation_norm(model: torch.nn.Module) -> dict[str, float]:
    """Per-site activation RMS (sqrt(energy / count)), keyed by the Linear's path.

    RMS, not L2: the window spans a whole optimizer step, so an L2 would scale with
    sqrt(batch x seq x gradient_accumulation_steps) and shift between runs whose
    batch schedule differs but whose model does not.
    """
    norms: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, ActivationStats):
            continue
        count = module.count.item()
        if count == 0:
            continue
        # the accumulator is a child of the Linear whose input it measures
        name = name.removeprefix("_orig_mod.")
        norms[name.rsplit(".", 1)[0]] = math.sqrt(module.energy.item() / count)
    return norms


def compute_weight_norms(model: torch.nn.Module) -> dict[str, float]:
    """
    Per-parameter L2 weight norms, keyed by parameter name. Stacked 3D expert
    weights are one norm over all experts, matching the grad-norm convention.
    """
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        name = name.removeprefix("_orig_mod.")
        norms[name] = param.detach().norm(2.0).item()
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


def compute_moe_maxvio(load_per_layer: list[torch.Tensor]) -> list[float]:
    """Per-layer MaxVio from accumulated expert load counts, in list order."""
    return [compute_maxvio(c) for c in load_per_layer]


def compute_moe_global_maxvio(load_per_layer: list[torch.Tensor]) -> list[float]:
    """Per-layer MaxVio of the batch-summed load from stacked ``[n_batch, E]`` counts."""
    return [compute_maxvio(loads.sum(0)) for loads in load_per_layer]


def compute_moe_batch_maxvio(load_per_layer: list[torch.Tensor]) -> list[float]:
    """Per-layer mean per-batch MaxVio from stacked ``[n_batch, E]`` counts."""
    return [
        statistics.mean([compute_maxvio(row) for row in loads])
        for loads in load_per_layer
    ]


# ---------------------------------------------------------------------------
# Weight spectral metrics (SVD)
# ---------------------------------------------------------------------------


def _gram_energy(weight: torch.Tensor) -> torch.Tensor:
    """Descending σ² spectrum via eigvalsh of the smaller Gram matrix (= svdvals(W)², no full SVD)."""
    w = weight.float()
    m, k = w.shape[-2], w.shape[-1]
    gram = w @ w.transpose(-2, -1) if m <= k else w.transpose(-2, -1) @ w
    eig = torch.linalg.eigvalsh(gram)  # ascending, (..., min(m, k))
    return eig.clamp_min(0).flip(-1)  # descending σ²


def _svd_metrics(weight: torch.Tensor) -> dict[str, float]:
    """srank / pr of a weight's σ² spectrum (averaged over experts if 3D)."""
    energy = _gram_energy(weight)  # (..., k) descending σ²
    total = energy.sum(-1)
    if (total == 0).any():
        return {"srank": 0.0, "pr": 0.0}
    srank = total / energy[..., 0]  # stable rank ‖W‖_F²/σ_max² (rank-1 collapse canary)
    pr = total.pow(2) / energy.pow(2).sum(-1)  # participation ratio (Σσ²)²/Σσ⁴
    return {"srank": srank.mean().item(), "pr": pr.mean().item()}


def compute_weight_svd_metrics(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Spectral metrics (srank/pr) per 2D/3D weight, keyed by param name (rope/embeddings skipped)."""
    metrics: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        if param.ndim not in (2, 3) or not param.is_floating_point():
            continue
        name = name.removeprefix("_orig_mod.")
        if name.startswith("rope.") or "emb" in name:
            continue
        metrics[name] = _svd_metrics(param.detach())
    return metrics


def compute_grad_svd_metrics(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Spectral metrics (srank/pr) per 2D/3D weight gradient (rope/embeddings skipped)."""
    metrics: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        if param.grad is None or param.ndim not in (2, 3):
            continue
        name = name.removeprefix("_orig_mod.")
        if name.startswith("rope.") or "emb" in name:
            continue
        metrics[name] = _svd_metrics(param.grad)
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
# Quantization error
# ---------------------------------------------------------------------------


def _quantization_metrics(src_sq, err_sq, under, numel, nonzero, grouped):
    """Reported metrics from one operand's accumulated sums, as 0-dim fp32 tensors.

    Empty experts are excluded: they have no error to report. Grouped operands also
    report the worst expert, because averaging over experts hides the single bad one
    that per-expert scales exist to prevent.

    A former range_ratio (amax/median) was dropped: the median is a sort, per expert
    per operand, and it dominated the cost of a logged step.
    """
    # clamps match the pre-vectorized form, which clamped each norm before the ratio
    sqnr = 20.0 * torch.log10(
        src_sq.sqrt().clamp_min(EPS) / err_sq.sqrt().clamp_min(EPS)
    )
    valid = (numel > 0).to(src_sq.dtype)
    n_valid = valid.sum().clamp_min(1.0)
    # over the values that could underflow: a source element that was already exactly
    # zero cannot, so dividing by numel would under-report by the operand's zero
    # fraction -- large for grads (ignore_index) and relu activations.
    informative = nonzero.clamp_min(1.0)
    underflow_rate = under / informative
    metrics = {
        "sqnr": (sqnr * valid).sum() / n_valid,
        "underflow_rate": (underflow_rate * valid).sum() / n_valid,
    }
    if not grouped:
        return metrics
    # min over a dB quantity, max over the rate: both are defined over the whole
    # range, unlike the max/min ratio a signed dB value would make meaningless
    keep = valid > 0
    # sqnr needs signal to measure: an expert with numel > 0 but nonzero == 0 (it
    # received tokens that were all exactly zero) has no error to report, and its
    # sqnr of 0.0 would otherwise drag sqnr_min down next to healthy experts.
    has_signal = nonzero > 0
    zeros = torch.zeros_like(underflow_rate)
    metrics["sqnr_min"] = torch.where(
        has_signal, sqnr, torch.full_like(sqnr, float("inf"))
    ).min()
    metrics["underflow_rate_max"] = torch.where(keep, underflow_rate, zeros).max()
    return metrics


def compute_quantization_metrics(model: torch.nn.Module) -> dict[str, float]:
    """Read a window's quantization accumulators as `<metric>/<site>` floats.

    Phase-agnostic, like `compute_activation_norm`: the caller prefixes
    `train-quant/` or `val-quant/`, because only the caller knows which window it
    just closed. Keyed off each accumulator's own `key`, not its module path: which
    GEMM and operand it belongs to is not something a path can express.

    A site whose accumulated numel is all zero was never folded this window --
    e.g. a dgrad/wgrad operand during a no_grad eval pass -- and is dropped rather
    than reported: its sqnr would read 0.0, the same value that means "noise equals
    signal". The emptiness flag rides the same single stack-and-sync as the metric
    values below, so this costs no extra device sync.
    """
    metrics = {}
    has_data = {}
    for module in model.modules():
        if not isinstance(module, QuantizationStats):
            continue
        sums = [getattr(module, name) for name in QuantizationStats.FIELDS]
        keep = (module.numel.sum() > 0).float()
        for name, value in _quantization_metrics(*sums, module.grouped).items():
            key = f"{name}/{module.key}"
            metrics[key] = value
            has_data[key] = keep
    if not metrics:
        return {}
    keys = list(metrics)
    combined = torch.stack(
        [metrics[k] for k in keys] + [has_data[k] for k in keys]
    ).tolist()  # one sync
    n = len(keys)
    return {k: v for k, v, keep in zip(keys, combined[:n], combined[n:]) if keep}


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
