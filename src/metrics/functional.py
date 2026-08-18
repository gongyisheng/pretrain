"""Stateless metric computations."""

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
    """Return L2 gradient norms keyed by parameter name."""
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        # torch.compile wraps model in OptimizedModule, prepending "_orig_mod."
        name = name.removeprefix("_orig_mod.")
        norms[name] = param.grad.data.norm(2.0).item()
    return norms


def compute_activation_norm(model: torch.nn.Module) -> dict[str, float]:
    """Return activation RMS keyed by the parent linear path."""
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
    """Return L2 weight norms keyed by parameter name."""
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
    """Return an MoE layer's maximum relative expert-load violation."""
    counts = expert_counts.float()
    mean = counts.mean()
    if mean == 0:
        return 0.0
    return ((counts.max() - mean) / mean).item()


def compute_moe_maxvio(load_per_layer: list[torch.Tensor]) -> list[float]:
    """Per-layer MaxVio from accumulated expert load counts, in list order."""
    return [compute_maxvio(c) for c in load_per_layer]


def compute_moe_global_maxvio(load_per_layer: list[torch.Tensor]) -> list[float]:
    """Return batch-summed MaxVio for each MoE layer."""
    return [compute_maxvio(loads.sum(0)) for loads in load_per_layer]


def compute_moe_batch_maxvio(load_per_layer: list[torch.Tensor]) -> list[float]:
    """Return mean per-batch MaxVio for each MoE layer."""
    return [
        statistics.mean([compute_maxvio(row) for row in loads])
        for loads in load_per_layer
    ]


# ---------------------------------------------------------------------------
# Weight spectral metrics (SVD)
# ---------------------------------------------------------------------------


def _gram_energy(weight: torch.Tensor) -> torch.Tensor:
    """Return descending squared singular values without a full SVD."""
    w = weight.float()
    m, k = w.shape[-2], w.shape[-1]
    gram = w @ w.transpose(-2, -1) if m <= k else w.transpose(-2, -1) @ w
    eig = torch.linalg.eigvalsh(gram)  # ascending, (..., min(m, k))
    return eig.clamp_min(0).flip(-1)  # descending σ²


def _svd_metrics(weight: torch.Tensor) -> dict[str, float]:
    """Return stable rank and participation ratio, averaged for 3D weights."""
    energy = _gram_energy(weight)  # (..., k) descending σ²
    total = energy.sum(-1)
    if (total == 0).any():
        return {"srank": 0.0, "pr": 0.0}
    srank = total / energy[..., 0]  # stable rank ‖W‖_F²/σ_max² (rank-1 collapse canary)
    pr = total.pow(2) / energy.pow(2).sum(-1)  # participation ratio (Σσ²)²/Σσ⁴
    return {"srank": srank.mean().item(), "pr": pr.mean().item()}


def compute_weight_svd_metrics(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Return spectral metrics for non-embedding 2D and 3D weights."""
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
    """Return spectral metrics for non-embedding 2D and 3D weight gradients."""
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
    """Clone trainable parameters for `compute_param_step_norm`."""
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def compute_param_step_norm(
    model: torch.nn.Module, snapshot: list[torch.Tensor]
) -> float:
    """Return the L2 norm of parameter changes since a snapshot."""
    total_sq: torch.Tensor | None = None
    params = (p for p in model.parameters() if p.requires_grad)
    for p, p_before in zip(params, snapshot, strict=True):
        sq = (p.detach() - p_before).pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    return total_sq.sqrt().item() if total_sq is not None else 0.0


def _aggregate_state_norm(
    optimizer: torch.optim.Optimizer, *, key: str
) -> float | None:
    """Return the L2 norm of optimizer buffers for `key`, or `None`."""
    total_sq: torch.Tensor | None = None
    for state in optimizer.state.values():
        buf = state.get(key)
        if buf is None:
            continue
        sq = buf.pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    return total_sq.sqrt().item() if total_sq is not None else None


def compute_momentum_norm(optimizer: torch.optim.Optimizer) -> float | None:
    """Return the L2 norm of first-moment buffers, or `None`."""
    return _aggregate_state_norm(optimizer, key="exp_avg")


def compute_variance_norm(optimizer: torch.optim.Optimizer) -> float | None:
    """Return the L2 norm of second-moment buffers, or `None`."""
    return _aggregate_state_norm(optimizer, key="exp_avg_sq")


# ---------------------------------------------------------------------------
# Quantization error
# ---------------------------------------------------------------------------


def _quantization_metrics(src_sq, err_sq, under, numel, nonzero, grouped):
    """Return quantization metrics from one operand's accumulated statistics."""
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
    """Return quantization metrics for populated sites as `<metric>/<site>` floats."""
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
    """Return mean, median, maximum, and minimum, or `{}` if empty."""
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
    ignore_index: int = -100,
    exclude_id: int | None = None,
) -> tuple[int, int]:
    """Return correct and eligible prediction counts."""
    preds = logits.argmax(dim=-1)
    mask = labels != ignore_index
    if exclude_id is not None:
        mask = mask & (labels != exclude_id)
    correct = (preds[mask] == labels[mask]).sum().item()
    total = int(mask.sum().item())
    return int(correct), total


def compute_decoded_byte_len(tokenizer, token_ids: list[int]) -> int:
    """Return the UTF-8 byte length of decoded token IDs."""
    return len(tokenizer.decode(token_ids, skip_special_tokens=True).encode("utf-8"))


def compute_bytes_per_token(tokenizer, texts: list[str]) -> float:
    """Return UTF-8 bytes per token across `texts`."""
    n_bytes = 0
    n_tokens = 0
    for t in texts:
        n_bytes += len(t.encode("utf-8"))
        n_tokens += len(tokenizer.encode(t, add_special_tokens=False).ids)
    if n_tokens == 0:
        raise ValueError("no tokens produced; corpus may be empty")
    return n_bytes / n_tokens


def compute_perplexity(loss: float, cap: float = 1e6) -> float:
    """Return capped `exp(loss)`."""
    return min(float(torch.exp(torch.tensor(loss))), cap)


def compute_bits_per_byte(loss: float, tokens_per_byte: float) -> float:
    """Convert a nats/token loss to bits/byte: loss * tokens_per_byte / ln(2)."""
    return loss * tokens_per_byte / math.log(2)


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------


def count_parameters(config: TrainConfig) -> dict[str, int]:
    """Return total, non-embedding, and active parameter counts."""
    return TransformerLM.compute_parameters(config.model, config.max_seq_len)


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def compute_flops_per_token(config: TrainConfig) -> int:
    """Return total training FLOPs per token."""
    fwd_total = TransformerLM.compute_flops(config.model, config.max_seq_len)
    return fwd_total * 3  # fwd + bwd (2x); no activation recomputation


# ---------------------------------------------------------------------------
# GPU peak FLOPS estimation (for MFU)
# ---------------------------------------------------------------------------


def estimate_gpu_peak_flops(device: str) -> float | None:
    """Estimate bf16/fp16 GPU peak FLOPS, or `None` if unknown."""
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
