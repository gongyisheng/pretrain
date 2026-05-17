"""Tokenizer training (BPE and SuperBPE).

Two methods are supported:
- "bpe": standard byte-level BPE (HuggingFace tokenizers).
- "superbpe": two-stage curriculum from arXiv:2503.13423.

All trainers emit a HuggingFace-compatible tokenizer.json under save_path/.
Runtime loading lives in src/data/tokenizer.py.
"""

import itertools
import os
import time
from typing import Callable, Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
import wandb

from src.eval.tokenizer import _bytes_per_token
from src.utils.config import TrainConfig


class TokenizerWandbLogger:
    """Thin wandb wrapper. Handles init/log/finish only — no metric computation."""

    def __init__(self, config: TrainConfig, enabled: bool = True):
        self.enabled = enabled
        if self.enabled:
            log = config.logging
            wandb.init(
                project=log.wandb_project,
                name=log.wandb_run_name,
                group=log.wandb_group or None,
                config=config.to_dict(),
            )
            wandb.define_metric("vocab_size")
            wandb.define_metric("bytes_per_token", step_metric="vocab_size")

    def log(self, metrics: dict) -> None:
        if self.enabled:
            wandb.log(metrics)

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()


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


class TokenizerTrainer:
    """Train a tokenizer from a text iterable. Dispatches by `config.data.tokenizer_train_method`.

    Method-specific kwargs live in `config.data.tokenizer_train_method_kwargs` and are
    consumed by the matching `_train_*` method (e.g. `transition_size`,
    `max_superword_words`, `eval_num_docs` for "superbpe"). W&B identity comes
    from `config.logging`; the `wandb_enabled` flag is constructor-controlled
    (default True), matching the model `Trainer`.

    The constructor validates the method/kwargs, creates the save directory,
    and builds `self.logger` and `self.metrics` so the inner `_train_*` methods
    only need to focus on the training loop.
    """

    _SPECIAL_TOKENS = ("<|endoftext|>",)

    def __init__(self, config: TrainConfig, wandb_enabled: bool = True):
        self.config = config
        self.wandb_enabled = wandb_enabled
        self.vocab_size = config.model.vocab_size
        self.save_path = config.data.tokenizer_path
        self.train_method = config.data.tokenizer_train_method
        self.train_method_kwargs = dict(config.data.tokenizer_train_method_kwargs)
        self.eval_every = config.data.tokenizer_train_eval_every
        self.eval_num_docs = self.train_method_kwargs.get("eval_num_docs", 1000)

        if self.train_method not in ("bpe", "superbpe"):
            raise ValueError(
                f"unknown method: {self.train_method!r}; expected 'bpe' or 'superbpe'"
            )

        # SuperBPE-specific knobs, validated up front so `_train_superbpe` can
        # assume they're well-formed.
        self.transition_size: int | None = None
        self.max_superword_words: int = 4
        if self.train_method == "superbpe":
            ts = self.train_method_kwargs.get("transition_size")
            if ts is None or not (0 < ts <= self.vocab_size):
                raise ValueError(
                    f"method='superbpe' requires 0 < transition_size <= vocab_size; "
                    f"got transition_size={ts}, vocab_size={self.vocab_size}"
                )
            self.transition_size = ts
            self.max_superword_words = self.train_method_kwargs.get(
                "max_superword_words", 4
            )

        os.makedirs(self.save_path, exist_ok=True)
        self.logger = TokenizerWandbLogger(config, enabled=wandb_enabled)
        self.metrics = TokenizerMetricsTracker()

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
        self.metrics.eval_texts = eval_texts

        # Build a factory that yields fresh training streams (skipping the
        # eval slice). Tiny-dataset fallback: if the underlying source is
        # exhausted by the eval slice, train on eval_texts itself.
        def make_train_iter() -> Iterable[str]:
            it = itertools.islice(dataset_iter(), self.eval_num_docs, None)
            try:
                first = next(it)
                return itertools.chain([first], it)
            except StopIteration:
                return iter(eval_texts)

        if self.train_method == "bpe":
            tokenizer = self._train_bpe(make_train_iter)
        else:
            tokenizer = self._train_superbpe(make_train_iter)

        tokenizer.save(os.path.join(self.save_path, "tokenizer.json"))
        print(
            f"{self.train_method.upper()} tokenizer saved to {self.save_path}/ "
            f"(vocab_size={tokenizer.get_vocab_size()})"
        )
        self.logger.finish()
        return tokenizer

    # ---- BPE ----

    def _train_bpe(self, make_train_iter: Callable[[], Iterable[str]]) -> Tokenizer:
        from src.data.bpe import BpeTrainer

        def curve_cb(vocab_size, vocab, merges):
            if not self.logger.enabled:
                return
            tmp = Tokenizer(models.BPE(vocab=vocab, merges=merges))
            tmp.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tmp.decoder = decoders.ByteLevel()
            self.logger.log(self.metrics.build_train_log_dict(tmp, vocab_size))

        bpe = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self._SPECIAL_TOKENS,
            pretokenizer="bpe",
            progress_callback=curve_cb if self.logger.enabled else None,
            progress_every=self.eval_every,
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
        from src.data.bpe import BpeTrainer

        def curve_cb(vocab_size, vocab, merges):
            if not self.logger.enabled:
                return
            tmp = Tokenizer(models.BPE(vocab=vocab, merges=merges))
            tmp.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False
            )
            tmp.decoder = decoders.ByteLevel()
            self.logger.log(self.metrics.build_train_log_dict(tmp, vocab_size))

        # ---- Stage 1: standard BPE with paper-default whitespace pretokenization ----
        t0 = time.perf_counter()
        stage1 = BpeTrainer(
            vocab_size=self.transition_size,
            special_tokens=self._SPECIAL_TOKENS,
            pretokenizer="bpe",
            progress_callback=curve_cb if self.logger.enabled else None,
            progress_every=self.eval_every,
        )
        stage1_vocab, stage1_merges = stage1.train(make_train_iter)

        stage1_tok = Tokenizer(models.BPE(vocab=stage1_vocab, merges=stage1_merges))
        stage1_tok.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False
        )
        stage1_tok.decoder = decoders.ByteLevel()
        stage1_tok.save(os.path.join(self.save_path, "bpe_tokenizer.json"))
        print(
            f"SuperBPE stage 1 done: vocab_size={len(stage1_vocab)} "
            f"({time.perf_counter() - t0:.2f}s)"
        )

        # ---- Stage 2: byte-level chunking + SuperBPE filters ----
        t0 = time.perf_counter()

        def superbpe_filter(a: str, b: str, merged: str) -> bool:
            if ":Ġ" in merged:
                return False
            word_count = merged.count("Ġ") + (0 if merged.startswith("Ġ") else 1)
            return word_count <= self.max_superword_words

        stage2 = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self._SPECIAL_TOKENS,
            pretokenizer="bytelevel",
            initial_vocab=stage1_vocab,
            initial_merges=stage1_merges,
            merge_filter=superbpe_filter,
            progress_callback=curve_cb if self.logger.enabled else None,
            progress_every=self.eval_every,
        )
        vocab, merges = stage2.train(make_train_iter)
        print(
            f"SuperBPE stage 2 done: vocab_size={len(vocab)} "
            f"({time.perf_counter() - t0:.2f}s)"
        )

        final = Tokenizer(models.BPE(vocab=vocab, merges=merges))
        final.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False
        )
        final.decoder = decoders.ByteLevel()
        return final
