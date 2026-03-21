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
from src.utils.config import TrainConfig


class Trainer:
    def __init__(self, config: TrainConfig, wandb_enabled: bool = True, resume_from: str = None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.step = 0
        self.loss_history = []

        # Seed for reproducibility
        self._seed(42)

        # Model
        self.model = build_model(config).to(self.device)
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

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.training.batch_size,
            shuffle=True, num_workers=config.data.num_workers, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.training.batch_size,
            shuffle=False, num_workers=config.data.num_workers, pin_memory=True,
        )

        # Optimizer & scheduler
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # Mixed precision
        self.use_amp = config.training.mixed_precision != "no" and self.device == "cuda"
        self.amp_dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # Activation checkpointing
        if config.training.activation_checkpointing:
            for block in self.model.blocks:
                block._original_forward = block.forward
                def make_ckpt_forward(b):
                    def ckpt_forward(x):
                        return torch.utils.checkpoint.checkpoint(b._original_forward, x, use_reentrant=False)
                    return ckpt_forward
                block.forward = make_ckpt_forward(block)

        # Tokenizer
        self.tokenizer = load_tokenizer(config.data.tokenizer_path)

        # Logger
        self.logger = WandbLogger(config, enabled=wandb_enabled)

        # Checkpoint dir
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)

        # Debug
        self.spike_debugger = SpikeDebugger(config.debug.spike, config.training.checkpoint_dir)

        # Resume
        if resume_from:
            self._load_checkpoint(resume_from)

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def train(self):
        cfg = self.config.training
        self.model.train()

        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        t_last_log = time.time()
        tokens_since_log = 0

        pbar = tqdm(total=cfg.max_steps, initial=self.step, desc="[train]", dynamic_ncols=True)
        while self.step < cfg.max_steps:
            self.optimizer.zero_grad()

            for micro_step in range(cfg.gradient_accumulation_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast(self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / cfg.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()
                tokens_since_log += x.numel()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.step += 1
            self.total_tokens += self.tokens_per_step
            self.loss_history.append(accum_loss)
            pbar.update(1)

            # Logging
            if self.step % self.config.logging.log_every == 0:
                elapsed = time.time() - t_last_log
                tokens_per_sec = tokens_since_log / elapsed if elapsed > 0 else 0
                lr = self.optimizer.param_groups[0]["lr"]
                flops = 6 * self.n_non_emb_params * self.total_tokens
                self.logger.log({
                    "train/loss": accum_loss,
                    "train/perplexity": min(float(torch.exp(torch.tensor(accum_loss))), 1e6),
                    "train/flops": flops,
                    "train/total_tokens": self.total_tokens,
                    "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "tokens_per_sec": tokens_per_sec,
                }, step=self.step)
                pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}", tok_s=f"{tokens_per_sec:.0f}")
                t_last_log = time.time()
                tokens_since_log = 0

            # Debug: spike detection
            if self.config.debug.spike.enabled:
                self.spike_debugger.on_step(
                    grad_norm, self.step, self.model,
                    save_checkpoint_fn=lambda: self._save_checkpoint(suffix="spike") or
                        os.path.join(cfg.checkpoint_dir, f"step_{self.step}_spike.pt"),
                )

            accum_loss = 0.0

            # Evaluation
            if self.step % cfg.eval_every == 0:
                self._evaluate()

            # Checkpoint
            if self.step % cfg.checkpoint_every == 0:
                self._save_checkpoint()

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
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
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
        self.model.eval()
        # <|endoftext|> (token 0) acts as BOS, prompting the model to start a new document
        idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            # truncate context to max_seq_len if generation grows long
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits = self.model(idx_cond)
            logits = logits[:, -1, :]  # take last token's logits
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # sample from distribution
            idx = torch.cat([idx, next_token], dim=1)
        token_ids = idx[0].tolist()
        generated_text = self.tokenizer.decode(token_ids)
        self.logger.log_text("generated_text", generated_text, step=self.step)

    def _save_checkpoint(self, suffix: str = ""):
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
        torch.save(checkpoint, path)
        print(f"[ckpt] saved to {path}")

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
