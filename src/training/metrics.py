"""Training metrics: per-step tracking, log_dict assembly, per-layer grad norms."""

import re
from collections import defaultdict

import torch

from src.utils.config import TrainConfig

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


class MetricsTracker:
    """Tracks and assembles training metrics for W&B logging."""

    def __init__(self, config: TrainConfig, n_non_emb_params: int, device: str):
        self.config = config
        self.n_non_emb_params = n_non_emb_params
        self.device = device
        self.is_moe = config.model.arch == "qwen3_moe"
        if self.is_moe:
            self._aux_floor = config.model.n_layers * config.model.n_experts_per_token

        self._grad_clip_steps = 0
        self._steps_since_log = 0
        self._skipped_steps = 0
        self._gpu_peak_flops = self._estimate_gpu_peak_flops()

    # ------------------------------------------------------------------
    # Per-step tracking
    # ------------------------------------------------------------------

    def on_step(
        self,
        grad_norm_val: float,
        grad_clip: float,
        scaler_enabled: bool,
        scale_before: float,
        scale_after: float,
    ):
        """Call after grad clip + scaler.step/update on every training step."""
        if grad_norm_val > grad_clip:
            self._grad_clip_steps += 1
        self._steps_since_log += 1
        if scaler_enabled and scale_after < scale_before:
            self._skipped_steps += 1

    # ------------------------------------------------------------------
    # Log dict assembly
    # ------------------------------------------------------------------

    def build_log_dict(
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
            "train/perplexity": min(float(torch.exp(torch.tensor(loss))), 1e6),
            "train/flops": 6 * self.n_non_emb_params * total_tokens,
            "train/total_tokens": total_tokens,
            # optim
            "optim/lr": lr,
            "optim/grad_clip_ratio": self._grad_clip_steps / max(self._steps_since_log, 1),
            "optim/loss_scale": scaler.get_scale() if scaler.is_enabled() else 1.0,
            "optim/skipped_steps": self._skipped_steps,
            # perf
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/step_time_ms": step_time_ms,
            # grad norm
            "grad_norm/total": grad_norm,
        }

        # MFU
        if self._gpu_peak_flops and tokens_per_sec > 0:
            flops_per_sec = 6 * self.n_non_emb_params * tokens_per_sec
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

    # ------------------------------------------------------------------
    # Per-layer gradient norms
    # ------------------------------------------------------------------

    @staticmethod
    def compute_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
        """Compute L2 gradient norms grouped by layer/component.

        Groups:
            grad_norm/embedding          - token_emb, pos_emb
            grad_norm/blocks.{i}.attn    - ln1 + attention params for block i
            grad_norm/blocks.{i}.ffn     - ln2 + FFN params for block i
            grad_norm/blocks.{i}.router  - MoE router params (MoE only)
            grad_norm/final_norm         - ln_f

        Each group norm is sqrt(sum of squared per-param norms).
        """
        group_sq: dict[str, float] = defaultdict(float)

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            sq = param.grad.data.norm(2.0).item() ** 2

            if name.startswith(("token_emb.", "pos_emb.", "lm_head.")):
                group_sq["grad_norm/embedding"] += sq
            elif name.startswith("ln_f."):
                group_sq["grad_norm/final_norm"] += sq
            elif (m := _BLOCK_RE.match(name)):
                idx = m.group(1)
                rest = m.group(2)
                if rest.startswith(("ln1.", "attn.")):
                    group_sq[f"grad_norm/blocks.{idx}.attn"] += sq
                elif rest.startswith("ffn.router."):
                    group_sq[f"grad_norm/blocks.{idx}.router"] += sq
                elif rest.startswith(("ln2.", "ffn.")):
                    group_sq[f"grad_norm/blocks.{idx}.ffn"] += sq
                else:
                    group_sq[f"grad_norm/blocks.{idx}.other"] += sq
            else:
                group_sq["grad_norm/other"] += sq

        return {key: val**0.5 for key, val in group_sq.items()}

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
