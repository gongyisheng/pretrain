import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.checkpoint
from tqdm import tqdm

from src.model.registry import build_model
from src.data.dataset import PretrainDataset
from src.data.tokenizer import load_tokenizer
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.logger import WandbLogger
from src.training.debug import SpikeDebugger
from src.training.metrics import MetricsTracker
from src.utils.config import TrainConfig
from src.kernel.triton.cross_entropy import triton_cross_entropy
from src.kernel.torch.cross_entropy import torch_cross_entropy


class Trainer:
    def __init__(self, config: TrainConfig, wandb_enabled: bool = True, resume_from: str = None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.step = 0
        self.loss_history = []
        self.debug_metrics = []  # per-step: {"step", "tokens_per_sec", "total_tokens"}

        # Enable TF32 for matmuls on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


        # Seed for reproducibility
        self._seed(42)

        # Backend selection: "torch" (torch.compile) or "triton" (custom kernels)
        from src.model.components import set_backend
        set_backend(config.training.backend)
        self._cross_entropy = triton_cross_entropy if config.training.backend == "triton" else torch_cross_entropy

        if config.training.intra_doc_mask and config.training.backend == "triton":
            raise ValueError(
                "intra_doc_mask is not supported with the triton backend. "
                "Use training.backend=torch instead."
            )
        self.intra_doc_mask = config.training.intra_doc_mask
        self.eot_token_id = config.data.eot_token_id

        # Model
        self.model = build_model(config).to(self.device)
        self.is_moe = (config.model.arch == "qwen3_moe")
        n_params = sum(p.numel() for p in self.model.parameters())
        self.n_non_emb_params = n_params - sum(
            p.numel() for name, p in self.model.named_parameters()
            if "emb" in name
        )
        self.tokens_per_step = (
            config.training.batch_size
            * config.training.gradient_accumulation_steps
            * config.max_seq_len
        )
        self.total_tokens = 0
        print(f"Model: {config.model.arch} | {n_params / 1e6:.1f}M params "
              f"({self.n_non_emb_params / 1e6:.1f}M non-embedding) | device={self.device}")

        # Data
        train_path = os.path.join(config.data.data_dir, "train.bin")
        val_path = os.path.join(config.data.data_dir, "val.bin")
        self.train_dataset = PretrainDataset(train_path, config.max_seq_len)
        self.val_dataset = PretrainDataset(val_path, config.max_seq_len)

        nw = config.data.num_workers
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.training.batch_size,
            shuffle=True, num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.training.batch_size,
            shuffle=False, num_workers=nw, pin_memory=True,
            persistent_workers=nw > 0,
        )

        # Optimizer & scheduler
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # Mixed precision
        self.use_amp = config.training.mixed_precision != "no" and self.device == "cuda"
        self.amp_dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))

        if self.is_moe and config.model.moe_expert_capacity_factor is None:
            # Dynamic capacity uses .item() — can't compile the full model
            for block in self.model.blocks:
                block.attn = torch.compile(block.attn)
        else:
            self.model = torch.compile(self.model)
        print(f"[trainer] backend={config.training.backend}")

        # Activation checkpointing
        if config.training.activation_checkpointing:
            if self.is_moe:
                print("[trainer] WARNING: activation_checkpointing is not supported for MoE; skipping.")
            else:
                for block in self.model.blocks:
                    block._original_forward = block.forward
                    def make_ckpt_forward(b):
                        def ckpt_forward(x, **kwargs):
                            return torch.utils.checkpoint.checkpoint(b._original_forward, x, use_reentrant=False, **kwargs)
                        return ckpt_forward
                    block.forward = make_ckpt_forward(block)

        # Tokenizer (may be None when tokenizer_path is empty, e.g. in tests)
        self.tokenizer = load_tokenizer(config.data.tokenizer_path) if config.data.tokenizer_path else None

        # Metrics
        self.metrics = MetricsTracker(config, self.n_non_emb_params, self.device)

        # Logger
        self.logger = WandbLogger(config, enabled=wandb_enabled)

        # Debug
        self.spike_debugger = SpikeDebugger(config.debug.spike, config.training.checkpoint_dir)

        # Checkpoint dir
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)

        # Resume
        if resume_from:
            self._load_checkpoint(resume_from)

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _next_batch(self, train_iter):
        try:
            return next(train_iter), train_iter
        except StopIteration:
            train_iter = iter(self.train_loader)
            return next(train_iter), train_iter

    def _build_doc_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-token document IDs from EOT token positions.

        doc_ids[b, i] = number of EOT tokens seen in x[b, 0..i-1].
        The EOT token itself belongs to the document it ends.
        """
        is_eot = (x == self.eot_token_id)
        doc_ids = torch.zeros_like(x)
        if x.shape[1] > 1:
            doc_ids[:, 1:] = is_eot[:, :-1].long().cumsum(dim=1)
        return doc_ids

    def train(self):
        cfg = self.config.training
        self.model.train()

        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        t_last_log = time.time()
        tokens_since_log = 0

        # Data prefetch stream for overlapping H2D transfer with compute
        prefetch_stream = torch.cuda.Stream() if self.device == "cuda" else None

        # Deferred loss: keep previous step's loss tensor to read while next step runs
        prev_loss_tensor = None

        stop_at = self.config.debug.max_steps if self.config.debug.max_steps > 0 else cfg.max_steps
        pbar = tqdm(total=stop_at, initial=self.step, desc="[train]", dynamic_ncols=True)
        while self.step < stop_at:
            self.optimizer.zero_grad(set_to_none=True)

            # Read previous step's loss NOW (GPU has been computing since we launched it)
            # This overlaps the .item() sync with the current step's data prefetch
            if prev_loss_tensor is not None:
                accum_loss = prev_loss_tensor.item()

            # Accumulate loss as tensor to avoid CUDA sync every micro-step
            accum_loss_tensor = torch.zeros(1, device=self.device)

            # Prefetch first batch
            (x_cpu, y_cpu), train_iter = self._next_batch(train_iter)
            if prefetch_stream is not None:
                with torch.cuda.stream(prefetch_stream):
                    x = x_cpu.to(self.device, non_blocking=True)
                    y = y_cpu.to(self.device, non_blocking=True)
            else:
                x, y = x_cpu.to(self.device), y_cpu.to(self.device)

            for micro_step in range(cfg.gradient_accumulation_steps):
                # Wait for current batch transfer to complete
                if prefetch_stream is not None:
                    torch.cuda.current_stream().wait_stream(prefetch_stream)

                # Start prefetching next batch while computing
                if micro_step < cfg.gradient_accumulation_steps - 1:
                    (next_x_cpu, next_y_cpu), train_iter = self._next_batch(train_iter)
                    if prefetch_stream is not None:
                        with torch.cuda.stream(prefetch_stream):
                            next_x = next_x_cpu.to(self.device, non_blocking=True)
                            next_y = next_y_cpu.to(self.device, non_blocking=True)
                    else:
                        next_x, next_y = next_x_cpu.to(self.device), next_y_cpu.to(self.device)

                with torch.amp.autocast(self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                    doc_ids = self._build_doc_ids(x) if self.intra_doc_mask else None
                    if self.is_moe:
                        logits, aux_loss = self.model(x, doc_ids=doc_ids)
                        ce_loss = self._cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                        loss = ce_loss + self.config.model.moe_aux_loss_coef * aux_loss
                    else:
                        logits = self.model(x, doc_ids=doc_ids)
                        loss = self._cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / cfg.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accum_loss_tensor += loss.detach()
                tokens_since_log += x.numel()

                # Swap to prefetched batch
                if micro_step < cfg.gradient_accumulation_steps - 1:
                    x, y = next_x, next_y

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            # Optimizer step
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.metrics.on_step(
                grad_norm_val, 
                cfg.grad_clip,
                self.scaler.is_enabled(), 
                scale_before, 
                self.scaler.get_scale()
            )
            self.scheduler.step()

            # Defer loss sync to next iteration (save tensor, read later)
            prev_loss_tensor = accum_loss_tensor

            self.step += 1
            self.total_tokens += self.tokens_per_step
            self.loss_history.append(accum_loss)
            pbar.update(1)

            # Logging (uses accum_loss from *previous* step — 1-step delay, no impact on training)
            if self.step % self.config.logging.log_every == 0:
                elapsed = time.time() - t_last_log
                tokens_per_sec = tokens_since_log / elapsed if elapsed > 0 else 0
                lr = self.optimizer.param_groups[0]["lr"]
                log_dict = self.metrics.build_log_dict(
                    loss=accum_loss, total_tokens=self.total_tokens, lr=lr,
                    grad_norm=grad_norm_val, tokens_per_sec=tokens_per_sec,
                    elapsed=elapsed, model=self.model, scaler=self.scaler,
                    aux_loss=aux_loss.item() if self.is_moe else None,
                )
                self.logger.log(log_dict, step=self.step)
                pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}", tok_s=f"{tokens_per_sec:.0f}")
                self.debug_metrics.append({
                    "step": self.step,
                    "tokens_per_sec": tokens_per_sec,
                    "total_tokens": self.total_tokens,
                })
                t_last_log = time.time()
                tokens_since_log = 0

            # Debug: spike detection
            if self.config.debug.spike.enabled:
                self.spike_debugger.on_step(
                    grad_norm, self.step, self.model,
                    save_checkpoint_fn=lambda extra=None: self._save_checkpoint(suffix="spike", extra=extra),
                    clip_value=cfg.grad_clip,
                )

            # Evaluation
            if self.step % cfg.eval_every == 0:
                self._evaluate()

            # Checkpoint
            if self.step % cfg.checkpoint_every == 0:
                self._save_checkpoint()

        # Final loss sync for the last step
        if prev_loss_tensor is not None:
            accum_loss = prev_loss_tensor.item()
            self.loss_history[-1] = accum_loss

        pbar.close()
        self.logger.finish()

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.config.training.eval_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                doc_ids = self._build_doc_ids(x) if self.intra_doc_mask else None
                if self.is_moe:
                    logits, _ = self.model(x, doc_ids=doc_ids)
                else:
                    logits = self.model(x, doc_ids=doc_ids)
                loss = self._cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ppl = min(float(torch.exp(torch.tensor(avg_loss))), 1e6)
        self.logger.log({"val/loss": avg_loss, "val/perplexity": ppl}, step=self.step)
        print(f"\n[eval] val_loss={avg_loss:.4f} | val_ppl={ppl:.2f}")

        self._generate_sample()
        self.model.train()

    @torch.no_grad()
    def _generate_sample(self, max_new_tokens: int = 50):
        """Generate a short text sample for qualitative monitoring."""
        if self.tokenizer is None:
            return
        self.model.eval()
        # <|endoftext|> (token 0) acts as BOS, prompting the model to start a new document
        idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            # truncate context to max_seq_len if generation grows long
            idx_cond = idx[:, -self.config.max_seq_len:]
            if self.is_moe:
                logits, _ = self.model(idx_cond)
            else:
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]  # take last token's logits
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # sample from distribution
            idx = torch.cat([idx, next_token], dim=1)
        token_ids = idx[0].tolist()
        generated_text = self.tokenizer.decode(token_ids)
        self.logger.log_text("generated_text", generated_text, step=self.step)

    def _save_checkpoint(self, suffix: str = "", extra: dict = None):
        name = f"step_{self.step}_{suffix}.pt" if suffix else f"step_{self.step}.pt"
        path = os.path.join(self.config.training.checkpoint_dir, name)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "grad_scaler": self.scaler.state_dict(),
            "step": self.step,
            "config": self.config.to_dict(),
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        print(f"[ckpt] saved to {path}")
        return path

    def _load_checkpoint(self, path: str):
        print(f"Resuming from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["grad_scaler"])
        self.step = checkpoint["step"]
        self.total_tokens = self.step * self.tokens_per_step

        rng = checkpoint.get("rng_states", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.random.set_rng_state(rng["torch"].cpu().to(torch.uint8).contiguous())
        if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"].cpu().to(torch.uint8).contiguous())

        print(f"Resumed at step {self.step}")
