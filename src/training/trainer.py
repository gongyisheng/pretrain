import itertools
import math
import os
import random
import time
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.checkpoint
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tqdm import tqdm

from src.model import build_model
from src.data.bpe import BpeTrainer
from src.data.dataset import PretrainDataset, SFTDataset
from src.data.tokenizer import load_tokenizer
from src.training.fp8 import maybe_convert_to_fp8
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.metrics import MetricsTracker, TokenizerMetricsTracker
from src.training.loss import LOSS_REGISTRY, compute_loss
from src.utils.config import TrainConfig
from src.utils.metric_utils import count_correct
from src.utils.tracking_utils import WandbLogger
from src.utils.masking_utils import (
    build_causal_attention_mask,
    build_intra_doc_attention_mask,
)


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

        if config.training.loss_fn not in LOSS_REGISTRY:
            raise ValueError(
                f"unknown loss_fn {config.training.loss_fn!r}; "
                f"expected one of {sorted(LOSS_REGISTRY)}"
            )

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
        self.is_moe = config.model.mlp_cls == "moe"

        # Data
        if not os.path.isdir(config.data.data_dir):
            raise FileNotFoundError(
                f"data_dir not found: {config.data.data_dir!r}; "
                f"run scripts/preprocess_data.py first"
            )
        if config.task == "pretrain":
            train_path = os.path.join(config.data.data_dir, "train.bin")
            val_path = os.path.join(config.data.data_dir, "val.bin")
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
            train_path = os.path.join(config.data.data_dir, "train.bin")
            val_path = os.path.join(config.data.data_dir, "val.bin")
            sft_kwargs = dict(
                seq_len=config.max_seq_len,
                vocab_size=config.model.vocab_size,
                packing=config.data.packing,
                eot_token_id=self.eot_token_id,
                pad_token_id=self.pad_token_id,
            )
            self.train_dataset = SFTDataset(train_path, **sft_kwargs)
            self.val_dataset = SFTDataset(val_path, **sft_kwargs)
        else:
            raise ValueError(
                f"unknown task: {config.task!r}; expected 'pretrain' or 'sft'"
            )

        n_worker = config.data.num_workers
        g = torch.Generator()
        g.manual_seed(config.training.seed)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=True,
            persistent_workers=n_worker > 0,
            prefetch_factor=config.data.prefetch_factor if n_worker > 0 else None,
            generator=g,
            worker_init_fn=Trainer._worker_init_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.eval_batch_size,
            shuffle=False,
            num_workers=n_worker,
            pin_memory=True,
            persistent_workers=n_worker > 0,
            prefetch_factor=config.data.prefetch_factor if n_worker > 0 else None,
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

        # FP8: swap eligible nn.Linear modules to Float8Linear. Must run before
        # torch.compile so the tracer sees the swapped modules.
        maybe_convert_to_fp8(self.model, config)

        # Disable assert_indirect_indexing to avoid spurious CUDA assertions during
        # torchinductor autotuning, which may dispatch kernel test runs on a stream
        # that doesn't respect wait_stream(prefetch_stream), causing RoPE's
        # position_ids[i] lookup to read uninitialized GPU memory.
        import torch._inductor.config as inductor_config

        inductor_config.assert_indirect_indexing = False

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

        # Metrics and Logging
        self.logger = WandbLogger(config, enabled=wandb_enabled)
        self.metrics = MetricsTracker(config, self.device, logger=self.logger)
        self.metrics.print_model_summary()

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
        self.metrics.train_begin()

        # Data prefetch stream for overlapping H2D transfer with compute
        prefetch_stream = torch.cuda.Stream() if self.device == "cuda" else None

        # Deferred loss: keep previous step's loss tensor to read while next step runs
        prev_loss_tensor = None

        stop_at = cfg.early_stop if cfg.early_stop > 0 else cfg.max_steps
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
                    if self.config.training.intra_doc_masking:
                        mask_dtype = self.amp_dtype if self.use_amp else torch.float32
                        attn_mask = build_intra_doc_attention_mask(
                            position_ids,
                            self.device,
                            mask_dtype,
                            attn_implementation=self.config.model.attn_kwargs[
                                "attn_implementation"
                            ],
                        )
                    else:
                        B, S = position_ids.shape
                        attn_mask = build_causal_attention_mask(
                            B,
                            S,
                            self.device,
                            attn_implementation=self.config.model.attn_kwargs[
                                "attn_implementation"
                            ],
                        )
                    logits, aux_loss = self.model(
                        input_ids, position_ids=position_ids, attn_mask=attn_mask
                    )
                    ce_loss = compute_loss(
                        logits,
                        labels,
                        self.config.training.loss_fn,
                        label_smoothing=self.config.training.label_smoothing,
                    )
                    loss = ce_loss
                    if aux_loss is not None:
                        loss = (
                            loss
                            + self.config.model.mlp_kwargs["aux_loss_coef"] * aux_loss
                        )
                    loss = loss / cfg.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accum_loss_tensor += ce_loss.detach() / cfg.gradient_accumulation_steps

                # Swap to prefetched batch
                if micro_step < cfg.gradient_accumulation_steps - 1:
                    input_ids, position_ids, labels = (
                        next_input_ids,
                        next_position_ids,
                        next_labels,
                    )

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            )
            grad_norm_val = (
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )

            if not math.isfinite(accum_loss):
                raise RuntimeError(
                    f"Loss is {accum_loss} at step {self.step}, stopping training"
                )

            self.metrics.snapshot_pre_step(self.model, self.step)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.metrics.on_train_step(
                loss=accum_loss,
                grad_norm=grad_norm_val,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                scale_before=scale_before,
                aux_loss=aux_loss,
            )
            self.model.post_step()

            self.scheduler.step()

            # Defer loss sync to next iteration (save tensor, read later)
            prev_loss_tensor = accum_loss_tensor

            self.step += 1
            pbar.update(1)

            log_dict = self.metrics.log_train(
                step=self.step,
                model=self.model,
                optimizer=self.optimizer,
            )
            if log_dict is not None:
                pbar.set_postfix(
                    loss=f"{log_dict['train/loss']:.4f}",
                    lr=f"{log_dict['optim/lr']:.2e}",
                    tok_s=f"{log_dict['perf/tokens_per_sec']:.0f}",
                )

            # Evaluation
            if self.step % cfg.eval_every == 0:
                self.model.eval()
                self._evaluate()
                self.model.train()

            # Checkpoint
            if self.step % cfg.checkpoint_every == 0:
                self._save_checkpoint()

        pbar.close()
        self.logger.finish()

    def _forward_batch(self, batch):
        """Move a (input_ids, position_ids, labels) batch to device, build the
        attention mask, and run a forward pass under autocast. Returns
        (logits, labels, aux_loss). Loss is left to the caller — wrap it in
        the same autocast context to match training-time numerics."""
        input_ids = batch[0].to(self.device)
        position_ids = batch[1].to(self.device)
        labels = batch[2].to(self.device)
        with torch.amp.autocast(
            self.device, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            if self.config.training.intra_doc_masking:
                mask_dtype = self.amp_dtype if self.use_amp else torch.float32
                attn_mask = build_intra_doc_attention_mask(
                    position_ids,
                    self.device,
                    mask_dtype,
                    attn_implementation=self.config.model.attn_kwargs[
                        "attn_implementation"
                    ],
                )
            else:
                B, S = position_ids.shape
                attn_mask = build_causal_attention_mask(
                    B,
                    S,
                    self.device,
                    attn_implementation=self.config.model.attn_kwargs[
                        "attn_implementation"
                    ],
                )
            logits, aux_loss = self.model(
                input_ids, position_ids=position_ids, attn_mask=attn_mask
            )
        return logits, labels, aux_loss

    @torch.no_grad()
    def _evaluate(self):
        self.metrics.eval_begin()
        for i, batch in enumerate(self.val_loader):
            if (
                self.config.training.eval_steps > 0
                and i >= self.config.training.eval_steps
            ):
                break
            logits, labels, aux_loss = self._forward_batch(batch)
            with torch.amp.autocast(
                self.device, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                loss = compute_loss(logits, labels, self.config.training.loss_fn)
            self.metrics.on_eval_step(
                loss=loss.item(),
                logits=logits,
                labels=labels,
                model=self.model,
                aux_loss=aux_loss.item() if aux_loss is not None else None,
                tokenizer=self.tokenizer,
                eot_token_id=self.eot_token_id,
            )

        train_avg_acc = None
        if self.config.task == "sft" and self.config.training.eval_train:
            train_avg_acc = self._evaluate_train_acc()

        # Assembly + dispatch + the printed summary line all live in MetricsTracker.
        self.metrics.log_eval(step=self.step, train_avg_acc=train_avg_acc)

        if self.config.task == "pretrain":
            self._generate_sample()

    @torch.no_grad()
    def _evaluate_train_acc(self) -> float:
        """Run a single pass over the train loader and return overall accuracy.

        Only used in SFT mode (`config.task == "sft"`) when `training.eval_train`
        is true. Cheap because grokking train sets are small (~3k samples).
        Caller is responsible for setting the model to eval mode.
        """
        correct = 0
        total = 0
        for batch in self.train_loader:
            logits, labels, _ = self._forward_batch(batch)
            _correct, _total = count_correct(
                logits, labels, exclude_id=self.eot_token_id
            )
            correct += _correct
            total += _total
        return correct / total if total > 0 else 0.0

    @torch.no_grad()
    def _generate_sample(self, max_new_tokens: int = 50):
        """Generate a short text sample for qualitative monitoring.

        Caller is responsible for setting the model to eval mode.
        """
        if self.tokenizer is None:
            return
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
                attn_implementation=self.config.model.attn_kwargs[
                    "attn_implementation"
                ],
            )
            with torch.amp.autocast(
                self.device, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                logits, _ = self.model(
                    idx_cond, position_ids=pos_ids, attn_mask=attn_mask
                )
            logits = logits[:, -1, :]  # take last token's logits
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probs, num_samples=1
            )  # sample from distribution
            idx = torch.cat([idx, next_token], dim=1)
        token_ids = idx[0].tolist()
        generated_text = self.tokenizer.decode(token_ids)
        self.logger.log_text("val-sample/generations", generated_text, step=self.step)

    def _save_checkpoint(self):
        name = f"step_{self.step}.pt"
        path = os.path.join(self.config.training.checkpoint_dir, name)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "grad_scaler": self.scaler.state_dict(),
            "step": self.step,
            "total_tokens": self.metrics.total_tokens,
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
        self.metrics.total_tokens = checkpoint.get(
            "total_tokens", self.step * self.metrics.tokens_per_step
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


class TokenizerTrainer:
    """Train a tokenizer from a text iterable. Dispatches by
    `config.tokenizer_training.method`.

    Method-specific kwargs live in `config.tokenizer_training.method_kwargs` and
    are consumed by the matching `_train_*` method (e.g. `transition_size`,
    `max_superword_words`, `eval_num_docs` for "superbpe"). W&B identity comes
    from `config.logging`; `wandb_enabled` is constructor-controlled (default
    True), matching the model `Trainer`.

    Two independent cadences (both in merges, both must be multiples of the
    BpeTrainer's `progress_every`): `eval_every` logs (vocab_size,
    bytes_per_token) to W&B; `checkpoint_every` saves the partial tokenizer to
    `checkpoint_dir`.
    """

    _SPECIAL_TOKENS = ("<|endoftext|>",)

    def __init__(self, config: TrainConfig, wandb_enabled: bool = True):
        self.config = config
        self.wandb_enabled = wandb_enabled
        self.vocab_size = config.model.vocab_size
        self.train_method = config.tokenizer_training.method
        self.train_method_kwargs = dict(config.tokenizer_training.method_kwargs)
        self.eval_every = config.tokenizer_training.eval_every
        self.checkpoint_every = config.tokenizer_training.checkpoint_every
        self.eval_num_docs = self.train_method_kwargs["eval_num_docs"]

        if self.train_method not in ("bpe", "superbpe"):
            raise ValueError(
                f"unknown method: {self.train_method!r}; expected 'bpe' or 'superbpe'"
            )

        # Validate SuperBPE-specific knobs up front so `_train_superbpe` can
        # assume they're well-formed when it reads them from train_method_kwargs.
        if self.train_method == "superbpe":
            ts = self.train_method_kwargs.get("transition_size")
            if ts is None or not (0 < ts <= self.vocab_size):
                raise ValueError(
                    f"method='superbpe' requires 0 < transition_size <= vocab_size; "
                    f"got transition_size={ts}, vocab_size={self.vocab_size}"
                )

        os.makedirs(config.tokenizer_training.checkpoint_dir, exist_ok=True)

        self.logger = WandbLogger(config, enabled=wandb_enabled)
        self.metrics = TokenizerMetricsTracker(self.logger)
        self.eval_texts: list[str] = []

    # ---- Shared helpers ----

    def _make_train_iter(
        self,
        dataset_iter: Callable[[], Iterable[str]],
        eval_texts: list[str],
    ) -> Callable[[], Iterable[str]]:
        """Factory: returns a zero-arg callable that yields a fresh training
        stream skipping the eval slice. Tiny-dataset fallback: if the source
        is exhausted by the eval slice, train on `eval_texts` itself.
        """

        def _factory() -> Iterable[str]:
            it = itertools.islice(dataset_iter(), self.eval_num_docs, None)
            try:
                first = next(it)
                return itertools.chain([first], it)
            except StopIteration:
                return iter(eval_texts)

        return _factory

    def _progress_callback(
        self, use_regex: bool, checkpoint: bool
    ) -> Callable[[int, dict, list], None]:
        """Factory: returns a `BpeTrainer.progress_callback` that, on the
        coarser eval/checkpoint cadences, logs to W&B (every `eval_every`
        merges, when the logger is enabled) and/or saves the partial tokenizer
        to `save_path` (every `checkpoint_every` merges, when `checkpoint`).

        The native callback fires at `BpeTrainer.progress_every` cadence;
        `eval_every`/`checkpoint_every` must be multiples of it or the throttle
        never fires. `use_regex` selects the eval/save-time pretokenizer
        (True for "bpe", False for superbpe passes — matching the saved config).
        """

        def _callback(vocab_size: int, vocab: dict, merges: list) -> None:
            do_log = self.logger.enabled and vocab_size % self.eval_every == 0
            do_ckpt = checkpoint and vocab_size % self.checkpoint_every == 0
            if not (do_log or do_ckpt):
                return
            tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=use_regex
            )
            tok.decoder = decoders.ByteLevel()
            if do_log:
                self.metrics.log_train(tok, vocab_size, self.eval_texts)
            if do_ckpt:
                self._save_checkpoint(tok)

        return _callback

    def _save_checkpoint(self, tokenizer: Tokenizer) -> None:
        """Save the tokenizer to `checkpoint_dir/tokenizer.json`."""
        path = os.path.join(
            self.config.tokenizer_training.checkpoint_dir, "tokenizer.json"
        )
        tokenizer.save(path)

    @staticmethod
    def evaluate(
        tokenizer_path: str,
        text_iter: Iterable[str],
        batch_size: int = 1000,
    ) -> dict:
        """Evaluate a saved tokenizer's encoding efficiency on a stream of texts.

        Returns {n_docs, n_bytes, n_tokens, bytes_per_token, tokens_per_byte}.
        """
        tokenizer = load_tokenizer(tokenizer_path)
        n_docs = 0
        n_bytes = 0
        n_tokens = 0
        batch: list[str] = []
        for text in text_iter:
            batch.append(text)
            n_bytes += len(text.encode("utf-8"))
            n_docs += 1
            if len(batch) >= batch_size:
                encs = tokenizer.encode_batch(batch, add_special_tokens=False)
                n_tokens += sum(len(e.ids) for e in encs)
                batch = []
        if batch:
            encs = tokenizer.encode_batch(batch, add_special_tokens=False)
            n_tokens += sum(len(e.ids) for e in encs)
        if n_tokens == 0:
            raise ValueError("no tokens produced; corpus may be empty")
        if n_bytes == 0:
            raise ValueError("no bytes produced; corpus may be empty")
        return {
            "n_docs": n_docs,
            "n_bytes": n_bytes,
            "n_tokens": n_tokens,
            "bytes_per_token": n_bytes / n_tokens,
            "tokens_per_byte": n_tokens / n_bytes,
        }

    def train(self, dataset_iter: Callable[[], Iterable[str]]) -> Tokenizer:
        """Train and save a tokenizer.

        `dataset_iter` is a zero-arg callable that returns a fresh text
        iterable each time it's called. SuperBPE needs the corpus twice
        (stage 1 training + stage 2 re-encoding) and we refuse to materialize
        the whole stream in memory — so the caller must provide something
        replayable (a generator function, an iterable, etc.).
        """
        # Pull eval_texts from the front of a fresh stream.
        eval_texts = list(itertools.islice(dataset_iter(), self.eval_num_docs))
        if not eval_texts:
            raise ValueError("dataset_iter produced no text")
        self.eval_texts = eval_texts

        make_train_iter = self._make_train_iter(dataset_iter, eval_texts)

        if self.train_method == "bpe":
            tokenizer = self._train_bpe(make_train_iter)
        else:
            tokenizer = self._train_superbpe(make_train_iter)

        self._save_checkpoint(tokenizer)
        print(
            f"{self.train_method.upper()} tokenizer saved to "
            f"{self.config.tokenizer_training.checkpoint_dir}/ "
            f"(vocab_size={tokenizer.get_vocab_size()})"
        )
        self.logger.finish()
        return tokenizer

    # ---- BPE ----

    def _train_bpe(self, make_train_iter: Callable[[], Iterable[str]]) -> Tokenizer:
        bpe = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self._SPECIAL_TOKENS,
            pretokenizer="bpe",
            progress_callback=self._progress_callback(use_regex=True, checkpoint=True),
            show_progress=True,
            progress_desc="[bpe]",
        )
        vocab, merges = bpe.train(make_train_iter)

        tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer

    # ---- SuperBPE ----

    def _train_superbpe(
        self, make_train_iter: Callable[[], Iterable[str]]
    ) -> Tokenizer:
        transition_size = self.train_method_kwargs["transition_size"]
        max_superword_words = self.train_method_kwargs["max_superword_words"]

        # ---- Subword pass: standard BPE with paper-default whitespace pretokenization ----
        t0 = time.perf_counter()
        subword_bpe = BpeTrainer(
            vocab_size=transition_size,
            special_tokens=self._SPECIAL_TOKENS,
            pretokenizer="bpe",
            progress_callback=self._progress_callback(use_regex=False, checkpoint=False)
            if self.logger.enabled
            else None,
            show_progress=True,
            progress_desc="[superbpe][subword pass]",
        )
        subword_vocab, subword_merges = subword_bpe.train(make_train_iter)

        subword_tok = Tokenizer(models.BPE(vocab=subword_vocab, merges=subword_merges))
        subword_tok.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False
        )
        subword_tok.decoder = decoders.ByteLevel()
        subword_tok.save(
            os.path.join(
                self.config.tokenizer_training.checkpoint_dir, "subword_tokenizer.json"
            )
        )
        print(
            f"SuperBPE subword pass done: vocab_size={len(subword_vocab)} "
            f"({time.perf_counter() - t0:.2f}s)"
        )

        # ---- Superword pass: byte-level chunking + SuperBPE filters ----
        t0 = time.perf_counter()

        def superbpe_filter(a: str, b: str, merged: str) -> bool:
            if ":Ġ" in merged:
                return False
            word_count = merged.count("Ġ") + (0 if merged.startswith("Ġ") else 1)
            return word_count <= max_superword_words

        superword_bpe = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self._SPECIAL_TOKENS,
            pretokenizer="bytelevel",
            initial_vocab=subword_vocab,
            initial_merges=subword_merges,
            merge_filter=superbpe_filter,
            progress_callback=self._progress_callback(use_regex=False, checkpoint=True),
            show_progress=True,
            progress_desc="[superbpe][superword pass]",
        )
        vocab, merges = superword_bpe.train(make_train_iter)
        print(
            f"SuperBPE superword pass done: vocab_size={len(vocab)} "
            f"({time.perf_counter() - t0:.2f}s)"
        )

        final = Tokenizer(models.BPE(vocab=vocab, merges=merges))
        final.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False
        )
        final.decoder = decoders.ByteLevel()
        return final
