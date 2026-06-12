"""Pure metric computation — stateless functions, no logging, no state.

The bottom layer of the metrics stack. Everything here takes numbers/tensors
in and returns numbers/dicts out; it never touches W&B or the training loop.
`MetricsTracker` (the stateful assembler) and `src/eval/` offline evaluators
both build on these primitives.
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
    """Per-parameter L2 gradient norms, keyed by parameter name.

    Keys are the raw parameter names (e.g. blocks.0.attn.q_proj.weight); the
    caller applies any logging namespace (e.g. a "grad_norm/" prefix). The
    "_orig_mod." prefix added by torch.compile is stripped.
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
# Optimizer step diagnostics (||Δθ||, ||m||, ||v||)
# ---------------------------------------------------------------------------


def snapshot_params(model: torch.nn.Module) -> list[torch.Tensor]:
    """Detached clone of every trainable parameter, ordered by model.parameters().

    Use with compute_param_step_norm to measure ||θ_after - θ_before||.
    """
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def compute_param_step_norm(
    model: torch.nn.Module, snapshot: list[torch.Tensor]
) -> float:
    """L2 norm of the parameter delta vs. an earlier snapshot.

    Accumulates the squared deltas in a single device tensor and syncs once
    (one .item()), rather than per parameter.
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
    """L2 norm across all `state[p][key]` buffers, or None if no buffer exists.

    Accumulates in a single device tensor and syncs once, not per buffer.
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
    """L2 norm across optimizer first-moment buffers.

    Reads `state[p]["exp_avg"]` for every param with state. Works for both
    Lion (single momentum buffer) and AdamW (first moment). Returns None if no
    such buffer exists (e.g. before the first optimizer.step()), so the caller
    can skip logging the metric.
    """
    return _aggregate_state_norm(optimizer, key="exp_avg")


def compute_variance_norm(optimizer: torch.optim.Optimizer) -> float | None:
    """L2 norm across optimizer second-moment buffers.

    Reads `state[p]["exp_avg_sq"]` for every param with state — AdamW's running
    average of squared gradients. Returns None for Lion (no second moment) and
    before the first optimizer.step(), so the caller can skip logging it.
    """
    return _aggregate_state_norm(optimizer, key="exp_avg_sq")


# ---------------------------------------------------------------------------
# Generic reducers
# ---------------------------------------------------------------------------


def compute_statistics(values: list[float]) -> dict[str, float]:
    """{mean, median, max, min} over a non-empty list of scalars.

    Generic reducer for probes that summarize a population (e.g. per-sample
    p_correct -> min/mean/max). Returns an empty dict for empty input.
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
    """Count positions where argmax(logits) == labels.

    Masks out `ignore_index` (loss-ignored positions) and, when given,
    `exclude_id` (e.g. the EOT token, which is trivially predictable in SFT).
    Returns (correct, total) as python ints.
    """
    preds = logits.argmax(dim=-1)
    mask = labels != ignore_index
    if exclude_id is not None:
        mask = mask & (labels != exclude_id)
    correct = (preds[mask] == labels[mask]).sum().item()
    total = int(mask.sum().item())
    return int(correct), total


def compute_decoded_byte_len(tokenizer, token_ids: list[int]) -> int:
    """UTF-8 byte length of `token_ids` decoded back to text.

    Used to convert a token-space loss to bits-per-byte: counting the bytes the
    target tokens decode to gives the tokens/byte ratio for that conversion.
    Special tokens are dropped (skip_special_tokens=True).
    """
    return len(tokenizer.decode(token_ids, skip_special_tokens=True).encode("utf-8"))


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


def count_parameters(model: torch.nn.Module, config: TrainConfig) -> dict[str, int]:
    """Total, non-embedding, and active-non-embedding parameter counts.

    For dense models active_non_emb == non_emb. For MoE, active_non_emb
    replaces each layer's full expert-FFN params with the k-activated subset
    (what actually runs per token).
    """
    total = sum(p.numel() for p in model.parameters())
    non_emb = total - sum(
        p.numel() for name, p in model.named_parameters() if "emb" in name
    )

    mc = config.model
    if mc.mlp_cls == "moe":
        mk = mc.mlp_kwargs
        n_exp = mk["n_experts"]
        k = mk["n_experts_per_token"]
        d_ff = mk.get("intermediate_size") or 4 * mc.d_model
        bias = mk.get("bias", False)
        expert_ffn_per_layer = n_exp * 3 * d_ff * mc.d_model
        active_ffn_per_layer = k * 3 * d_ff * mc.d_model
        if bias:
            expert_ffn_per_layer += n_exp * (2 * d_ff + mc.d_model)
            active_ffn_per_layer += k * (2 * d_ff + mc.d_model)
        active_non_emb = (
            non_emb
            - mc.n_layers * expert_ffn_per_layer
            + mc.n_layers * active_ffn_per_layer
        )
    else:
        active_non_emb = non_emb

    return {"total": total, "non_emb": non_emb, "active_non_emb": active_non_emb}


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def compute_flops_per_token(config: TrainConfig) -> dict[str, int]:
    """Per-token training FLOPs: ``fwd_total`` (from the model's own
    ``compute_flops``) and ``total`` (fwd × backward multiplier, 4 if
    activation_checkpointing else 3).
    """
    fwd_total = TransformerLM.compute_flops(config.model, config.max_seq_len)
    backward_mult = 4 if config.training.activation_checkpointing else 3
    return {"fwd_total": fwd_total, "total": fwd_total * backward_mult}


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
