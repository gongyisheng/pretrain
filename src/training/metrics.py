"""MetricsTracker — the stateful assembly layer between pure metric math
(`src/utils/metric_utils.py`) and dispatch (`src/utils/tracking_utils.py`).

It owns everything that has state across a training run: per-window counters,
log-cadence timing, cached optimizer step-norms, running token totals
(tokens_per_step / total_tokens), and the eval accumulators. It calls
`metric_utils` for arithmetic and a logger (WandbLogger) for dispatch; the
trainer feeds it raw numbers and never assembles a log_dict itself.

Setup (once):
    print_model_summary(model)            # startup banner: param counts + device
    train_begin()                         # reset the log-window timer

Lifecycle per optimizer step:
    snapshot_pre_step(model, step)        # before optimizer.step() (snapshots
                                          #   only on pre-log steps)
    ...scaler.step / update...
    on_step(loss=, grad_norm=, ...)       # update counters + total_tokens,
                                          #   cache step-norms
    log_train(step=, model=, optimizer=)  # on cadence: assemble + dispatch,
                                          #   else None

Eval:
    eval_begin()
    eval_step(loss=, logits=, labels=, ...)   # per batch
    log_eval(step=)                            # finalize + dispatch + print
"""

import time

import torch
from tokenizers import Tokenizer

from src.eval.tokenizer import _bytes_per_token
from src.utils import metric_utils
from src.utils.config import TrainConfig
from src.utils.tracking_utils import WandbLogger


class MetricsTracker:
    """Tracks and assembles training/eval metrics, then dispatches to W&B."""

    def __init__(
        self,
        config: TrainConfig,
        device: str,
        logger: WandbLogger,
    ):
        self.config = config
        self.device = device
        self.logger = logger

        self.is_moe = config.model.mlp_cls == "moe"
        if self.is_moe:
            self._aux_floor = (
                config.model.n_layers * config.model.mlp_kwargs["n_experts_per_token"]
            )

        self._flops_per_token = metric_utils.compute_flops_per_token(config)
        self._gpu_peak_flops = metric_utils.estimate_gpu_peak_flops(device)
        self.tokens_per_step = (
            config.training.batch_size
            * config.training.gradient_accumulation_steps
            * config.max_seq_len
        )

        # Per-window counters (reset every log_every steps).
        self._grad_clip_steps = 0
        self._steps_since_log = 0
        self._tokens_since_log = 0
        # Cumulative across the run (intentionally not reset).
        # total_tokens is public:
        # the trainer persists/restores it via the checkpoint.
        self._skipped_steps = 0
        self.total_tokens = 0
        self._t_last_log = time.time()

        self._last_loss = 0.0
        self._last_grad_norm = 0.0
        self._loss_scale = 1.0
        self._last_aux_loss: float | None = None
        self._param_snapshot: list[torch.Tensor] | None = None
        self._param_step_norm: float | None = None
        self._momentum_norm: float | None = None
        self._variance_norm: float | None = None

        # Eval accumulators (initialized by eval_begin).
        self._eval_loss_sum = 0.0
        self._eval_aux_sum = 0.0
        self._eval_n_batches = 0
        self._eval_bpb_tokens = 0
        self._eval_bpb_bytes = 0
        self._eval_acc_correct = 0
        self._eval_acc_total = 0

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------

    def print_model_summary(self, model: torch.nn.Module) -> None:
        """Print the one-line startup banner (param counts + device).

        MoE models report total params + active (k-expert) non-embedding params;
        dense models report total + non-embedding.
        """
        counts = metric_utils.count_parameters(model, self.config)
        label = f"{self.config.model.attn_cls}+{self.config.model.mlp_cls}"
        if self.is_moe:
            msg = (
                f"Model: {label} | {counts['total'] / 1e6:.1f}M total params "
                f"({counts['active_non_emb'] / 1e6:.1f}M active non-embedding) "
                f"| device={self.device}"
            )
        else:
            msg = (
                f"Model: {label} | {counts['total'] / 1e6:.1f}M params "
                f"({counts['non_emb'] / 1e6:.1f}M non-embedding) | device={self.device}"
            )
        print(msg)

    def train_begin(self) -> None:
        """Reset the log-window timer/counter. Call at the top of train() so
        the first window's tokens/sec excludes construction (and resume) time.
        """
        self._t_last_log = time.time()
        self._tokens_since_log = 0

    def snapshot_pre_step(self, model: torch.nn.Module, step: int) -> None:
        """Cache θ before optimizer.step() so on_step can compute ||Δθ||.

        Only snapshots on the step whose update will be logged next
        (``(step + 1) % log_every == 0``): the step-norm logged at a cadence
        boundary reflects only that final pre-log update, so snapshotting +
        diffing on every step would clone the whole model and sync per-param
        for a value that is then overwritten and discarded. No-op when
        config.logging.log_optimizer_step_norms is False.
        """
        log_every = self.config.logging.log_every
        if self.config.logging.log_optimizer_step_norms and (step + 1) % log_every == 0:
            self._param_snapshot = metric_utils.snapshot_params(model)
        else:
            self._param_snapshot = None

    def on_step(
        self,
        *,
        loss: float,
        grad_norm: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        scale_before: float,
        aux_loss: torch.Tensor | None = None,
    ) -> None:
        """
        Update window counters and cache the values log_train needs.
        """
        self._last_loss = loss
        self._last_grad_norm = grad_norm
        self._loss_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
        self._last_aux_loss = aux_loss.item() if aux_loss is not None else None

        if grad_norm > self.config.training.grad_clip:
            self._grad_clip_steps += 1
        self._steps_since_log += 1
        self._tokens_since_log += self.tokens_per_step
        self.total_tokens += self.tokens_per_step
        if scaler.is_enabled() and scaler.get_scale() < scale_before:
            self._skipped_steps += 1

        if self._param_snapshot is not None:
            self._param_step_norm = metric_utils.compute_param_step_norm(
                model, self._param_snapshot
            )
            self._momentum_norm = metric_utils.compute_momentum_norm(optimizer)
            self._variance_norm = metric_utils.compute_variance_norm(optimizer)
            self._param_snapshot = None
        else:
            self._param_step_norm = None
            self._momentum_norm = None
            self._variance_norm = None

    def log_train(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float] | None:
        """On a log-cadence step: assemble the train log_dict, dispatch it to
        the logger, reset window counters, and return the dict (for the pbar).
        Off cadence: return None and do nothing. LR is read from the
        optimizer's first param group.
        """
        if step % self.config.logging.log_every != 0:
            return None

        lr = optimizer.param_groups[0]["lr"]
        now = time.time()
        elapsed = now - self._t_last_log
        tokens_per_sec = self._tokens_since_log / elapsed if elapsed > 0 else 0.0
        step_time_ms = (
            elapsed / self.config.logging.log_every * 1000 if elapsed > 0 else 0.0
        )

        loss = self._last_loss
        d: dict[str, float] = {
            # train
            "train/loss": loss,
            "train/flops": self._flops_per_token * self.total_tokens,
            "train/total_tokens": self.total_tokens,
            # optim
            "optim/lr": lr,
            "optim/grad_clip_ratio": self._grad_clip_steps
            / max(self._steps_since_log, 1),
            "optim/loss_scale": self._loss_scale,
            "optim/skipped_steps": self._skipped_steps,
            # perf
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/step_time_ms": step_time_ms,
            # grad norm
            "grad_norm/total": self._last_grad_norm,
        }

        # MoE aux loss (cached as a float in on_step)
        if self.is_moe and self._last_aux_loss is not None:
            d["train/aux_loss"] = self._last_aux_loss - self._aux_floor

        if self.config.task == "pretrain":
            d["train/perplexity"] = metric_utils.compute_perplexity(loss)

        # MFU
        if self._gpu_peak_flops and tokens_per_sec > 0:
            flops_per_sec = self._flops_per_token * tokens_per_sec
            d["perf/mfu"] = flops_per_sec / self._gpu_peak_flops

        # GPU memory
        if self.device == "cuda":
            d["perf/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            d["perf/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        # Optimizer step diagnostics (||Δθ||, ||m||, ||v||)
        if self._param_step_norm is not None:
            d["optim/param_step_norm"] = self._param_step_norm
        if self._momentum_norm is not None:
            d["optim/momentum_norm"] = self._momentum_norm
        if self._variance_norm is not None:
            d["optim/variance_norm"] = self._variance_norm

        # Per-layer gradient norms (grads still live: zero_grad runs next step)
        if self.config.logging.log_layer_grad_norms:
            for name, norm in metric_utils.compute_layer_grad_norms(model).items():
                d[f"grad_norm/{name}"] = norm

        self.logger.log(d, step=step)

        # Reset per-window counters/timer.
        self._grad_clip_steps = 0
        self._steps_since_log = 0
        self._tokens_since_log = 0
        self._t_last_log = now

        return d

    def eval_begin(self) -> None:
        """Reset eval accumulators before iterating the val loader."""
        self._eval_loss_sum = 0.0
        self._eval_aux_sum = 0.0
        self._eval_n_batches = 0
        self._eval_bpb_tokens = 0
        self._eval_bpb_bytes = 0
        self._eval_acc_correct = 0
        self._eval_acc_total = 0

    def eval_step(
        self,
        *,
        loss: float,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux_loss: float | None = None,
        tokenizer=None,
        eot_token_id: int | None = None,
    ) -> None:
        """Accumulate one eval batch's contributions (loss, accuracy, bpb).

        tokenizer (for bpb byte-counting) and eot_token_id (excluded from SFT
        accuracy) are passed per call rather than held on the tracker.
        """
        self._eval_loss_sum += loss
        self._eval_n_batches += 1
        if aux_loss is not None:
            self._eval_aux_sum += aux_loss

        if self.config.task == "sft":
            c, t = metric_utils.count_correct(logits, labels, exclude_id=eot_token_id)
            self._eval_acc_correct += c
            self._eval_acc_total += t

        if tokenizer is not None and self.config.task == "pretrain":
            # Count loss-contributing tokens; decode them and count UTF-8 bytes.
            keep = labels.reshape(-1) != -100
            target_ids = labels.reshape(-1)[keep].tolist()
            self._eval_bpb_tokens += len(target_ids)
            self._eval_bpb_bytes += metric_utils.compute_decoded_byte_len(
                tokenizer, target_ids
            )

    def log_eval(
        self, *, step: int, train_avg_acc: float | None = None
    ) -> dict[str, float]:
        """Finalize accumulators into the eval log_dict, dispatch it, print the
        human-readable summary line, and return the dict.

        Routing by config.task ('pretrain' vs 'sft') matches train-time keys:
        - "pretrain": val/loss, val/perplexity, val/bpb (when tokenizer present)
        - "sft": val/loss, val/val_acc, val/train_acc (when provided)
        """
        n = max(self._eval_n_batches, 1)
        avg_loss = self._eval_loss_sum / n
        avg_aux_loss = (self._eval_aux_sum / n) if self._eval_aux_sum > 0 else None
        tokens_per_byte = (
            self._eval_bpb_tokens / self._eval_bpb_bytes
            if self._eval_bpb_bytes > 0
            else None
        )
        avg_acc = (
            (self._eval_acc_correct / self._eval_acc_total)
            if (self.config.task == "sft" and self._eval_acc_total > 0)
            else None
        )

        d: dict[str, float] = {"val/loss": avg_loss}
        if self.config.task == "pretrain":
            d["val/perplexity"] = metric_utils.compute_perplexity(avg_loss)
            if tokens_per_byte is not None:
                d["val/bpb"] = metric_utils.compute_bits_per_byte(
                    avg_loss, tokens_per_byte
                )
        elif self.config.task == "sft":
            if avg_acc is not None:
                d["val/val_acc"] = avg_acc
            if train_avg_acc is not None:
                d["val/train_acc"] = train_avg_acc
        if self.is_moe and avg_aux_loss is not None:
            d["val/aux_loss"] = avg_aux_loss - self._aux_floor

        self.logger.log(d, step=step)
        print(self._format_eval_msg(d, avg_loss))
        return d

    @staticmethod
    def _format_eval_msg(d: dict[str, float], avg_loss: float) -> str:
        """Build the human-readable eval summary line from the log_dict."""
        msg = f"\n[eval] val_loss={avg_loss:.4f}"
        if "val/perplexity" in d:
            msg += f" | val_ppl={d['val/perplexity']:.2f}"
        if "val/bpb" in d:
            msg += f" | val_bpb={d['val/bpb']:.4f}"
        if "val/val_acc" in d:
            msg += f" | val_acc={d['val/val_acc']:.4f}"
        if "val/train_acc" in d:
            msg += f" | train_acc={d['val/train_acc']:.4f}"
        if "val/aux_loss" in d:
            msg += f" | val_aux_loss={d['val/aux_loss']:.4f}"
        return msg


class TokenizerMetricsTracker:
    """Builds W&B log dicts for tokenizer training. Pure compute — never talks
    to W&B; the trainer feeds outputs to the logger.

    `eval_texts` is set by the trainer after pulling the held-out slice from the
    dataset, so the tracker can be constructed before the corpus is consumed.
    """

    def __init__(self, eval_texts: list[str] | None = None):
        self.eval_texts: list[str] = eval_texts if eval_texts is not None else []

    def build_train_log_dict(
        self, tokenizer: Tokenizer, vocab_size: int
    ) -> dict[str, float]:
        """Assemble a per-step W&B log dict for a (partial or full) tokenizer."""
        return {
            "vocab_size": vocab_size,
            "bytes_per_token": _bytes_per_token(tokenizer, self.eval_texts),
        }

    def build_eval_log_dict(
        self, tokenizer: Tokenizer, vocab_size: int
    ) -> dict[str, float]:
        """Assemble a per-eval W&B log dict. Identical to build_train_log_dict
        for now; kept separate so the two can diverge as eval-only metrics
        (e.g. coverage, OOV rate) are added.
        """
        return {
            "vocab_size": vocab_size,
            "bytes_per_token": _bytes_per_token(tokenizer, self.eval_texts),
        }
