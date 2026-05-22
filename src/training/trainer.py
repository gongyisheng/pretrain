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
from src.data.dataset import PretrainDataset, SFTDataset
from src.data.tokenizer import load_tokenizer
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.logger import WandbLogger
from src.training.debug import SpikeDebugger
from src.training.metrics import MetricsTracker
from src.utils.config import TrainConfig
from src.training.loss import shift_inputs, compute_loss
from src.utils.masking_utils import (
    build_causal_attention_mask,
    build_intra_doc_attention_mask,
)


@torch.compile
def _cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


class Trainer:
    def __init__(
        self, config: TrainConfig, wandb_enabled: bool = True, resume_from: str = None
    ):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.step = 0

        # Enable TF32 for matmuls on Ampere+ GPUs
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Prefer deterministic CUDA algorithms
        if config.training.use_deterministic_algo:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Seed for reproducibility
        self._seed(config.training.seed)

        self._loss_fn = _cross_entropy

        # Tokenizer — loaded early to provide special token IDs to the dataset
        self.tokenizer = (
            load_tokenizer(config.data.tokenizer_path)
            if config.data.tokenizer_path
            else None
        )
        if self.tokenizer is not None:
            eot_id = self.tokenizer.token_to_id("<|endoftext|>")
            self.eot_token_id = eot_id if eot_id is not None else 0
        else:
            self.eot_token_id = 0  # fallback for tests without a real tokenizer
        # EOT doubles as the padding token (no dedicated pad token in the current tokenizer)
        self.pad_token_id = self.eot_token_id

        # Model
        self.model = build_model(config).to(self.device)
        self.is_moe = config.model.arch == "qwen3_moe"
        n_params = sum(p.numel() for p in self.model.parameters())
        self.n_non_emb_params = n_params - sum(
            p.numel() for name, p in self.model.named_parameters() if "emb" in name
        )
        # For MoE, FLOPs use active params (k experts activated) not total params.
        # Replace total expert FFN params with the k-active subset in the count.
        if self.is_moe:
            mc = config.model
            expert_ffn_per_layer = (
                mc.moe_n_experts * 3 * mc.moe_intermediate_size * mc.d_model
            )
            active_ffn_per_layer = (
                mc.moe_n_experts_per_token * 3 * mc.moe_intermediate_size * mc.d_model
            )
            if mc.mlp_bias:
                expert_ffn_per_layer += mc.moe_n_experts * (
                    2 * mc.moe_intermediate_size + mc.d_model
                )
                active_ffn_per_layer += mc.moe_n_experts_per_token * (
                    2 * mc.moe_intermediate_size + mc.d_model
                )
            self.n_active_non_emb_params = (
                self.n_non_emb_params
                - mc.n_layers * expert_ffn_per_layer
                + mc.n_layers * active_ffn_per_layer
            )
        else:
            self.n_active_non_emb_params = self.n_non_emb_params
        self.tokens_per_step = (
            config.training.batch_size
            * config.training.gradient_accumulation_steps
            * config.max_seq_len
        )
        self.total_tokens = 0
        if self.is_moe:
            print(
                f"Model: {config.model.arch} | {n_params / 1e6:.1f}M total params "
                f"({self.n_active_non_emb_params / 1e6:.1f}M active non-embedding) | device={self.device}"
            )
        else:
            print(
                f"Model: {config.model.arch} | {n_params / 1e6:.1f}M params "
                f"({self.n_non_emb_params / 1e6:.1f}M non-embedding) | device={self.device}"
            )

        # Data
        train_path = os.path.join(config.data.data_dir, "train.bin")
        val_path = os.path.join(config.data.data_dir, "val.bin")
        if config.task == "pretrain":
            dataset_kwargs = dict(
                packing=config.data.packing,
                eot_token_id=self.eot_token_id,
                pad_token_id=self.pad_token_id,
            )
            self.train_dataset = PretrainDataset(
                train_path,
                config.max_seq_len,
                config.model.vocab_size,
                **dataset_kwargs,
            )
            self.val_dataset = PretrainDataset(
                val_path,
                config.max_seq_len,
                config.model.vocab_size,
                **dataset_kwargs,
            )
        elif config.task == "sft":
            import json

            meta_path = os.path.join(config.data.data_dir, "meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            sft_kwargs = dict(
                vocab_size=config.model.vocab_size,
                question_len=meta["question_len"],
                answer_len=meta["answer_len"],
            )
            self.train_dataset = SFTDataset(train_path, **sft_kwargs)
            self.val_dataset = SFTDataset(val_path, **sft_kwargs)
        else:
            raise ValueError(
                f"unknown task: {config.task!r}; expected 'pretrain' or 'sft'"
            )

        nw = config.data.num_workers
        g = torch.Generator()
        g.manual_seed(config.training.seed)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
            prefetch_factor=1 if nw > 0 else None,
            generator=g,
            worker_init_fn=Trainer._worker_init_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
            prefetch_factor=1 if nw > 0 else None,
            generator=g,
            worker_init_fn=Trainer._worker_init_fn,
        )

        # Optimizer & scheduler
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # Mixed precision
        self.use_amp = config.training.mixed_precision != "no" and self.device == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if config.training.mixed_precision == "bf16"
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            enabled=(self.use_amp and self.amp_dtype == torch.float16)
        )

        # Disable assert_indirect_indexing to avoid spurious CUDA assertions during
        # torchinductor autotuning, which may dispatch kernel test runs on a stream
        # that doesn't respect wait_stream(prefetch_stream), causing RoPE's
        # position_ids[i] lookup to read uninitialized GPU memory.
        import torch._inductor.config as inductor_config

        inductor_config.assert_indirect_indexing = False

        if self.is_moe and config.model.moe_expert_capacity_factor is None:
            # Dynamic capacity uses .item() — can't compile the full model
            for block in self.model.blocks:
                block.attn = torch.compile(block.attn)
        else:
            self.model = torch.compile(self.model)

        # Activation checkpointing
        if config.training.activation_checkpointing:
            if self.is_moe:
                print(
                    "[trainer] WARNING: activation_checkpointing is not supported for MoE; skipping."
                )
            else:
                for block in self.model.blocks:
                    block._original_forward = block.forward

                    def make_ckpt_forward(b):
                        def ckpt_forward(x, **kwargs):
                            return torch.utils.checkpoint.checkpoint(
                                b._original_forward, x, use_reentrant=False, **kwargs
                            )

                        return ckpt_forward

                    block.forward = make_ckpt_forward(block)

        # Metrics
        self.metrics = MetricsTracker(config, self.n_active_non_emb_params, self.device)

        # Logger
        self.logger = WandbLogger(config, enabled=wandb_enabled)

        # Debug
        self.spike_debugger = SpikeDebugger(
            config.debug.spike, config.training.checkpoint_dir
        )

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

    @staticmethod
    def _worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)

    def _next_batch(self, train_iter):
        try:
            return next(train_iter), train_iter
        except StopIteration:
            train_iter = iter(self.train_loader)
            return next(train_iter), train_iter

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

        stop_at = (
            self.config.debug.max_steps
            if self.config.debug.max_steps > 0
            else cfg.max_steps
        )
        pbar = tqdm(
            total=stop_at, initial=self.step, desc="[train]", dynamic_ncols=True
        )
        while self.step < stop_at:
            self.optimizer.zero_grad(set_to_none=True)

            # Read previous step's loss NOW (GPU has been computing since we launched it)
            # This overlaps the .item() sync with the current step's data prefetch
            if prev_loss_tensor is not None:
                accum_loss = prev_loss_tensor.item()

            # Accumulate loss as tensor to avoid CUDA sync every micro-step
            accum_loss_tensor = torch.zeros(1, device=self.device)

            # Prefetch first batch
            batch_cpu, train_iter = self._next_batch(train_iter)
            input_ids_cpu, position_ids_cpu, labels_cpu = (
                batch_cpu[0],
                batch_cpu[1],
                batch_cpu[2],
            )
            if prefetch_stream is not None:
                with torch.cuda.stream(prefetch_stream):
                    input_ids = input_ids_cpu.to(self.device, non_blocking=True)
                    position_ids = position_ids_cpu.to(self.device, non_blocking=True)
                    labels = labels_cpu.to(self.device, non_blocking=True)
            else:
                input_ids = input_ids_cpu.to(self.device)
                position_ids = position_ids_cpu.to(self.device)
                labels = labels_cpu.to(self.device)

            for micro_step in range(cfg.gradient_accumulation_steps):
                # Wait for current batch transfer to complete
                if prefetch_stream is not None:
                    torch.cuda.current_stream().wait_stream(prefetch_stream)
                    # record_stream tells the caching allocator that the H2D'd
                    # tensors are in use on the current stream. Without it the
                    # allocator only tracks them against prefetch_stream and
                    # may free their storage as soon as the H2D copy finishes
                    # — while torch.compile's graph is still consuming them on
                    # the default stream. That race silently corrupts inputs
                    # and biases the gradient (worst case: ~+0.6 nats val gap).
                    input_ids.record_stream(torch.cuda.current_stream())
                    position_ids.record_stream(torch.cuda.current_stream())
                    labels.record_stream(torch.cuda.current_stream())

                # Start prefetching next batch while computing
                if micro_step < cfg.gradient_accumulation_steps - 1:
                    next_batch_cpu, train_iter = self._next_batch(train_iter)
                    next_input_ids_cpu, next_position_ids_cpu, next_labels_cpu = (
                        next_batch_cpu[0],
                        next_batch_cpu[1],
                        next_batch_cpu[2],
                    )
                    if prefetch_stream is not None:
                        with torch.cuda.stream(prefetch_stream):
                            next_input_ids = next_input_ids_cpu.to(
                                self.device, non_blocking=True
                            )
                            next_position_ids = next_position_ids_cpu.to(
                                self.device, non_blocking=True
                            )
                            next_labels = next_labels_cpu.to(
                                self.device, non_blocking=True
                            )
                    else:
                        next_input_ids = next_input_ids_cpu.to(self.device)
                        next_position_ids = next_position_ids_cpu.to(self.device)
                        next_labels = next_labels_cpu.to(self.device)

                with torch.amp.autocast(
                    self.device, dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    x = shift_inputs(input_ids)
                    if self.config.training.intra_doc_masking:
                        mask_dtype = self.amp_dtype if self.use_amp else torch.float32
                        attn_mask = build_intra_doc_attention_mask(
                            position_ids,
                            self.device,
                            mask_dtype,
                            attn_implementation=self.config.model.attn_implementation,
                        )
                    else:
                        B, S = position_ids.shape
                        attn_mask = build_causal_attention_mask(
                            B,
                            S,
                            self.device,
                            attn_implementation=self.config.model.attn_implementation,
                        )
                    logits, aux_loss = self.model(
                        x, position_ids=position_ids, attn_mask=attn_mask
                    )
                    loss = compute_loss(logits, labels, self._loss_fn)
                    if aux_loss is not None:
                        loss = loss + self.config.model.moe_aux_loss_coef * aux_loss
                    loss = loss / cfg.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accum_loss_tensor += loss.detach()
                tokens_since_log += x.numel()

                # Swap to prefetched batch
                if micro_step < cfg.gradient_accumulation_steps - 1:
                    input_ids, position_ids, labels = (
                        next_input_ids,
                        next_position_ids,
                        next_labels,
                    )

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            )
            grad_norm_val = (
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )

            # Optimizer step
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.metrics.on_step(
                accum_loss,
                self.step,
                grad_norm_val,
                cfg.grad_clip,
                self.scaler.is_enabled(),
                scale_before,
                self.scaler.get_scale(),
            )
            self.scheduler.step()

            # Defer loss sync to next iteration (save tensor, read later)
            prev_loss_tensor = accum_loss_tensor

            self.step += 1
            self.total_tokens += self.tokens_per_step
            pbar.update(1)

            # Logging (uses accum_loss from *previous* step — 1-step delay, no impact on training)
            if self.step % self.config.logging.log_every == 0:
                elapsed = time.time() - t_last_log
                tokens_per_sec = tokens_since_log / elapsed if elapsed > 0 else 0
                lr = self.optimizer.param_groups[0]["lr"]
                log_dict = self.metrics.build_train_log_dict(
                    loss=accum_loss,
                    total_tokens=self.total_tokens,
                    lr=lr,
                    grad_norm=grad_norm_val,
                    tokens_per_sec=tokens_per_sec,
                    elapsed=elapsed,
                    model=self.model,
                    scaler=self.scaler,
                    aux_loss=aux_loss.item() if aux_loss is not None else None,
                )
                self.logger.log(log_dict, step=self.step)
                pbar.set_postfix(
                    loss=f"{accum_loss:.4f}",
                    lr=f"{lr:.2e}",
                    tok_s=f"{tokens_per_sec:.0f}",
                )
                t_last_log = time.time()
                tokens_since_log = 0

            # Debug: spike detection
            if self.config.debug.spike.enabled:
                self.spike_debugger.on_step(
                    grad_norm,
                    self.step,
                    self.model,
                    save_checkpoint_fn=lambda extra=None: self._save_checkpoint(
                        suffix="spike", extra=extra
                    ),
                    clip_value=cfg.grad_clip,
                )

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
        total_aux_loss = 0.0
        n_batches = 0
        bpb_tokens = 0
        bpb_bytes = 0
        acc_correct = 0
        acc_total = 0

        for i, batch in enumerate(self.val_loader):
            if i >= self.config.training.eval_steps:
                break
            input_ids = batch[0].to(self.device)
            position_ids = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            with torch.amp.autocast(
                self.device, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                x = shift_inputs(input_ids)
                if self.config.training.intra_doc_masking:
                    mask_dtype = self.amp_dtype if self.use_amp else torch.float32
                    attn_mask = build_intra_doc_attention_mask(
                        position_ids,
                        self.device,
                        mask_dtype,
                        attn_implementation=self.config.model.attn_implementation,
                    )
                else:
                    B, S = position_ids.shape
                    attn_mask = build_causal_attention_mask(
                        B,
                        S,
                        self.device,
                        attn_implementation=self.config.model.attn_implementation,
                    )
                logits, aux_loss = self.model(
                    x, position_ids=position_ids, attn_mask=attn_mask
                )
                loss = compute_loss(logits, labels, self._loss_fn)
            total_loss += loss.item()
            if aux_loss is not None:
                total_aux_loss += aux_loss.item()
            if self.config.task == "sft":
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                acc_correct += (preds[mask] == labels[mask]).sum().item()
                acc_total += int(mask.sum().item())
            if self.tokenizer is not None and self.config.task == "pretrain":
                # Count loss-contributing tokens; decode them and count UTF-8 bytes.
                keep = labels.reshape(-1) != -100
                target_ids = labels.reshape(-1)[keep].tolist()
                bpb_tokens += len(target_ids)
                bpb_bytes += len(
                    self.tokenizer.decode(target_ids, skip_special_tokens=True).encode(
                        "utf-8"
                    )
                )
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_aux_loss = (
            (total_aux_loss / max(n_batches, 1)) if total_aux_loss > 0 else None
        )
        tokens_per_byte = bpb_tokens / bpb_bytes if bpb_bytes > 0 else None
        avg_acc = (
            (acc_correct / acc_total)
            if (self.config.task == "sft" and acc_total > 0)
            else None
        )
        train_avg_acc = None
        if self.config.task == "sft" and self.config.training.eval_train:
            train_avg_acc = self._evaluate_train_acc()
            self.model.eval()  # _evaluate_train_acc may have left model in eval; keep symmetry

        log_dict = self.metrics.build_eval_log_dict(
            avg_loss=avg_loss,
            avg_aux_loss=avg_aux_loss,
            tokens_per_byte=tokens_per_byte,
            avg_acc=avg_acc,
            train_avg_acc=train_avg_acc,
        )
        self.logger.log(log_dict, step=self.step)
        eval_msg = f"\n[eval] val_loss={avg_loss:.4f}"
        if "val/perplexity" in log_dict:
            eval_msg += f" | val_ppl={log_dict['val/perplexity']:.2f}"
        if "val/bpb" in log_dict:
            eval_msg += f" | val_bpb={log_dict['val/bpb']:.4f}"
        if "val/acc" in log_dict:
            eval_msg += f" | val_acc={log_dict['val/acc']:.4f}"
        if "train/acc" in log_dict:
            eval_msg += f" | train_acc={log_dict['train/acc']:.4f}"
        if "val/aux_loss" in log_dict:
            eval_msg += f" | val_aux_loss={log_dict['val/aux_loss']:.4f}"
        print(eval_msg)

        if self.config.task == "pretrain":
            self._generate_sample()
        self.model.train()

    @torch.no_grad()
    def _evaluate_train_acc(self) -> float:
        """Run a single pass over the train loader and return overall accuracy.

        Only used in SFT mode (`config.task == "sft"`) when `training.eval_train`
        is true. Cheap because grokking train sets are small (~3k samples).
        """
        self.model.eval()
        correct = 0
        total = 0
        for batch in self.train_loader:
            input_ids = batch[0].to(self.device)
            position_ids = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            with torch.amp.autocast(
                self.device, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                x = shift_inputs(input_ids)
                B, S = position_ids.shape
                attn_mask = build_causal_attention_mask(
                    B,
                    S,
                    self.device,
                    attn_implementation=self.config.model.attn_implementation,
                )
                logits, _ = self.model(
                    x, position_ids=position_ids, attn_mask=attn_mask
                )
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct += (preds[mask] == labels[mask]).sum().item()
            total += int(mask.sum().item())
        return correct / total if total > 0 else 0.0

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
            idx_cond = idx[:, -self.config.max_seq_len :]
            B, S = idx_cond.shape
            pos_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
            attn_mask = build_causal_attention_mask(
                B,
                S,
                self.device,
                attn_implementation=self.config.model.attn_implementation,
            )
            logits, _ = self.model(idx_cond, position_ids=pos_ids, attn_mask=attn_mask)
            logits = logits[:, -1, :]  # take last token's logits
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probs, num_samples=1
            )  # sample from distribution
            idx = torch.cat([idx, next_token], dim=1)
        token_ids = idx[0].tolist()
        generated_text = self.tokenizer.decode(token_ids)
        self.logger.log_text("val-sample/generations", generated_text, step=self.step)

    def _save_checkpoint(self, suffix: str = "", extra: dict = None):
        name = f"step_{self.step}_{suffix}.pt" if suffix else f"step_{self.step}.pt"
        path = os.path.join(self.config.training.checkpoint_dir, name)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "grad_scaler": self.scaler.state_dict(),
            "step": self.step,
            "total_tokens": self.total_tokens,
            "config": self.config.to_dict(),
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else None,
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
        # Prefer the saved token count so resumes with a different batch_size or ga
        # don't silently corrupt the running total. Fall back for old checkpoints.
        self.total_tokens = checkpoint.get(
            "total_tokens", self.step * self.tokens_per_step
        )

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
