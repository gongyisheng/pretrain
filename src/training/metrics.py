"""Training metrics: per-step tracking, log_dict assembly, per-layer grad norms."""

import math
import torch

from src.utils.config import TrainConfig


def compute_flops_per_token(config: TrainConfig) -> dict[str, int]:
    """Per-token training FLOPs broken down by component.

    Returns dict with: qkv_proj, o_proj, attn_matmul, ffn, norm, lm_head
    (each summed across n_layers where applicable), fwd_total, total. The
    total applies a backward multiplier of 4 if activation_checkpointing
    else 3. Attention matmul uses full max_seq_len (PaLM/nanoGPT convention,
    ignores causal masking and intra-doc masking; comparable to literature
    MFU numbers). Norms (pre-attn, pre-FFN, optional qk_norm, final
    pre-lm_head) are counted at ~3 FLOPs per element. Embedding lookup and
    RoPE are 0 FLOPs.
    """
    head_dim = config.model.d_model // config.model.n_heads

    # Attention projections (per layer per token)
    qkv_matmul = (
        2
        * config.model.d_model
        * (config.model.n_heads + 2 * config.model.n_kv_heads)
        * head_dim
    )
    qkv_bias = (
        (config.model.n_heads + 2 * config.model.n_kv_heads) * head_dim
        if config.model.attn_bias
        else 0
    )
    qkv_per_layer = qkv_matmul + qkv_bias

    o_matmul = 2 * config.model.d_model * config.model.d_model
    o_bias = config.model.d_model if config.model.attn_bias else 0
    o_per_layer = o_matmul + o_bias

    # Attention matmul: full max_seq_len (PaLM/nanoGPT convention, no causal
    # or intra-doc-mask discount; matches literature MFU). Per token per layer:
    # QK^T = 2*n_heads*head_dim*max_seq_len, AV = same -> sum = 4*...
    attn_matmul_per_layer = 4 * config.model.n_heads * head_dim * config.max_seq_len

    # FFN
    if config.model.moe_n_experts > 0:
        # Router: 2 * d * n_experts
        router = 2 * config.model.d_model * config.model.moe_n_experts
        d_ff = config.model.moe_intermediate_size
        k = config.model.moe_n_experts_per_token
        if config.model.mlp_gated:
            expert_matmul = 6 * config.model.d_model * d_ff
            expert_bias = (
                (2 * d_ff + config.model.d_model) if config.model.mlp_bias else 0
            )
        else:
            expert_matmul = 4 * config.model.d_model * d_ff
            expert_bias = (d_ff + config.model.d_model) if config.model.mlp_bias else 0
        ffn_per_layer = router + k * (expert_matmul + expert_bias)
    else:
        d_ff = config.model.intermediate_size
        if config.model.mlp_gated:
            ffn_matmul = 6 * config.model.d_model * d_ff
            ffn_bias = (2 * d_ff + config.model.d_model) if config.model.mlp_bias else 0
        else:
            ffn_matmul = 4 * config.model.d_model * d_ff
            ffn_bias = (d_ff + config.model.d_model) if config.model.mlp_bias else 0
        ffn_per_layer = ffn_matmul + ffn_bias

    # Norms (RMSNorm/LayerNorm ~3 FLOPs per element: x*x, sum, rsqrt, x*inv*gamma)
    # Per layer: pre-attn norm + pre-FFN norm. Plus optional qk_norm on Q/K.
    norm_ops_per_element = 3
    norm_per_layer = 2 * norm_ops_per_element * config.model.d_model
    if config.model.qk_norm:
        norm_per_layer += (
            norm_ops_per_element
            * (config.model.n_heads + config.model.n_kv_heads)
            * head_dim
        )
    final_norm = norm_ops_per_element * config.model.d_model  # before lm_head

    # LM head (once per token, not per layer)
    lm_head = 2 * config.model.d_model * config.model.vocab_size + (
        config.model.vocab_size if config.model.lm_head_bias else 0
    )

    qkv_proj = config.model.n_layers * qkv_per_layer
    o_proj = config.model.n_layers * o_per_layer
    attn_matmul = config.model.n_layers * attn_matmul_per_layer
    ffn = config.model.n_layers * ffn_per_layer
    norm = config.model.n_layers * norm_per_layer + final_norm
    fwd_total = qkv_proj + o_proj + attn_matmul + ffn + norm + lm_head

    backward_mult = 4 if config.training.activation_checkpointing else 3

    return {
        "qkv_proj": qkv_proj,
        "o_proj": o_proj,
        "attn_matmul": attn_matmul,
        "ffn": ffn,
        "norm": norm,
        "lm_head": lm_head,
        "fwd_total": fwd_total,
        "total": fwd_total * backward_mult,
    }


class MetricsTracker:
    """Tracks and assembles training metrics for W&B logging."""

    def __init__(self, config: TrainConfig, device: str):
        self.config = config
        self.device = device
        self.is_moe = config.model.arch == "qwen3_moe"
        if self.is_moe:
            self._aux_floor = (
                config.model.n_layers * config.model.moe_n_experts_per_token
            )

        self.flops_per_token = compute_flops_per_token(config)

        self._grad_clip_steps = 0
        self._steps_since_log = 0
        self._skipped_steps = 0
        self._gpu_peak_flops = self._estimate_gpu_peak_flops()

    # ------------------------------------------------------------------
    # Per-step tracking
    # ------------------------------------------------------------------

    def on_step(
        self,
        loss: float,
        step: int,
        grad_norm_val: float,
        grad_clip: float,
        scaler_enabled: bool,
        scale_before: float,
        scale_after: float,
    ):
        """Call after grad clip + scaler.step/update on every training step."""
        if not math.isfinite(loss):
            raise RuntimeError(f"Loss is {loss} at step {step}, stopping training")
        if grad_norm_val > grad_clip:
            self._grad_clip_steps += 1
        self._steps_since_log += 1
        if scaler_enabled and scale_after < scale_before:
            self._skipped_steps += 1

    # ------------------------------------------------------------------
    # Log dict assembly
    # ------------------------------------------------------------------

    def build_train_log_dict(
        self,
        *,
        loss: float,
        total_tokens: int,
        lr: float,
        grad_norm: float,
        tokens_per_sec: float,
        elapsed: float,
        model: torch.nn.Module,
        scaler: torch.amp.GradScaler,
        aux_loss: float | None = None,
    ) -> dict[str, float]:
        """Assemble full metrics dict for logging. Resets per-window counters."""
        log_every = self.config.logging.log_every
        step_time_ms = elapsed / log_every * 1000 if elapsed > 0 else 0

        d: dict[str, float] = {
            # train
            "train/loss": loss,
            "train/flops": self.flops_per_token["total"] * total_tokens,
            "train/total_tokens": total_tokens,
            # optim
            "optim/lr": lr,
            "optim/grad_clip_ratio": self._grad_clip_steps
            / max(self._steps_since_log, 1),
            "optim/loss_scale": scaler.get_scale() if scaler.is_enabled() else 1.0,
            "optim/skipped_steps": self._skipped_steps,
            # perf
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/step_time_ms": step_time_ms,
            # grad norm
            "grad_norm/total": grad_norm,
        }

        if self.config.task == "pretrain":
            d["train/perplexity"] = min(float(torch.exp(torch.tensor(loss))), 1e6)

        # MFU
        if self._gpu_peak_flops and tokens_per_sec > 0:
            flops_per_sec = self.flops_per_token["total"] * tokens_per_sec
            d["perf/mfu"] = flops_per_sec / self._gpu_peak_flops

        # GPU memory
        if self.device == "cuda":
            d["perf/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            d["perf/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        # MoE aux loss
        if self.is_moe and aux_loss is not None:
            d["train/aux_loss"] = aux_loss - self._aux_floor

        # Per-layer gradient norms
        if self.config.logging.log_layer_grad_norms:
            d.update(self.compute_layer_grad_norms(model))

        # Reset per-window counters
        self._grad_clip_steps = 0
        self._steps_since_log = 0

        return d

    def build_eval_log_dict(
        self,
        *,
        avg_loss: float,
        avg_aux_loss: float | None = None,
        tokens_per_byte: float | None = None,
        avg_acc: float | None = None,
        train_avg_acc: float | None = None,
    ) -> dict[str, float]:
        """Assemble validation metrics dict for logging.

        Routing is governed by `self.config.task`:
        - "pretrain": logs val/loss, val/perplexity, val/bpb (when tokenizer present).
        - "sft": logs val/loss, val/val_acc, and val/train_acc when train_avg_acc is given.
        """
        d: dict[str, float] = {"val/loss": avg_loss}
        if self.config.task == "pretrain":
            d["val/perplexity"] = min(float(torch.exp(torch.tensor(avg_loss))), 1e6)
            if tokens_per_byte is not None:
                d["val/bpb"] = avg_loss * tokens_per_byte / math.log(2)
        elif self.config.task == "sft":
            if avg_acc is not None:
                d["val/val_acc"] = avg_acc
            if train_avg_acc is not None:
                d["val/train_acc"] = train_avg_acc
        if self.is_moe and avg_aux_loss is not None:
            d["val/aux_loss"] = avg_aux_loss - self._aux_floor
        return d

    # ------------------------------------------------------------------
    # Per-layer gradient norms
    # ------------------------------------------------------------------

    @staticmethod
    def compute_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
        """Compute per-parameter L2 gradient norms.

        Each parameter (e.g. blocks.0.attn.q_proj.weight) gets its own key
        under grad_norm/. Automatically handles the _orig_mod. prefix added
        by torch.compile.
        """
        norms: dict[str, float] = {}

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            # torch.compile wraps model in OptimizedModule, prepending "_orig_mod."
            name = name.removeprefix("_orig_mod.")
            norms[f"grad_norm/{name}"] = param.grad.data.norm(2.0).item()

        return norms

    # ------------------------------------------------------------------
    # GPU peak FLOPS estimation
    # ------------------------------------------------------------------

    def _estimate_gpu_peak_flops(self) -> float | None:
        """Estimate GPU peak bf16/fp16 FLOPS for MFU calculation."""
        if self.device != "cuda":
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
