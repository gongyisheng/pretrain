"""Tokenizer training (BPE and SuperBPE).

Both methods drive the `BpeTrainer` in `src/data/bpe.py` (Python control
flow over a C++ hot-loop extension):
- "bpe":      one BpeTrainer call with whitespace + digit pretokenization.
- "superbpe": two BpeTrainer calls — a "subword pass" (same config as "bpe",
              merges stay within word boundaries) followed by a "superword
              pass" (pretokenizer="bytelevel", continues from the subword
              vocab/merges, merges may cross word boundaries; filtered by
              the `max_superword_words` cap and the ":Ġ" exclusion).
              Equivalent to stage 1 / stage 2 in arXiv:2503.13423.

All trainers emit a HuggingFace-compatible tokenizer.json under save_path/.
Runtime encode/decode loading lives in src/data/tokenizer.py.
"""

import itertools
import os
import time
from typing import Callable, Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
import wandb

from src.data.bpe import BpeTrainer
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

    def _wandb_curve_cb(self, use_regex: bool) -> Callable[[int, dict, list], None]:
        """Factory: returns a `BpeTrainer.progress_callback` that logs
        (vocab_size, bytes_per_token) to W&B.

        `use_regex` controls the eval-time pretokenizer:
        - True  for "bpe" mode (HF byte-level + default regex word splitting)
        - False for "superbpe" stage 1 and stage 2 (pure byte-level, no
                splitting — matches the saved tokenizer.json config)

        Caller is responsible for only invoking this when `logger.enabled`
        is True; the returned callback assumes it.
        """

        def _cb(vocab_size: int, vocab: dict, merges: list) -> None:
            tmp = Tokenizer(models.BPE(vocab=vocab, merges=merges))
            tmp.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=use_regex
            )
            tmp.decoder = decoders.ByteLevel()
            self.logger.log(self.metrics.build_train_log_dict(tmp, vocab_size))

        return _cb

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

        make_train_iter = self._make_train_iter(dataset_iter, eval_texts)

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
        bpe = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self._SPECIAL_TOKENS,
            pretokenizer="bpe",
            progress_callback=self._wandb_curve_cb(use_regex=True)
            if self.logger.enabled
            else None,
            progress_every=self.eval_every,
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
        # ---- Subword pass: standard BPE with paper-default whitespace pretokenization ----
        t0 = time.perf_counter()
        subword_bpe = BpeTrainer(
            vocab_size=self.transition_size,
            special_tokens=self._SPECIAL_TOKENS,
            pretokenizer="bpe",
            progress_callback=self._wandb_curve_cb(use_regex=False)
            if self.logger.enabled
            else None,
            progress_every=self.eval_every,
            show_progress=True,
            progress_desc="[superbpe][subword pass]",
        )
        subword_vocab, subword_merges = subword_bpe.train(make_train_iter)

        subword_tok = Tokenizer(models.BPE(vocab=subword_vocab, merges=subword_merges))
        subword_tok.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False
        )
        subword_tok.decoder = decoders.ByteLevel()
        subword_tok.save(os.path.join(self.save_path, "subword_tokenizer.json"))
        print(
            f"SuperBPE subword pass done: vocab_size={len(subword_vocab)} "
            f"({time.perf_counter() - t0:.2f}s)"
        )

        # ---- Superword pass: byte-level chunking + SuperBPE filters ----
        t0 = time.perf_counter()

        superword_bpe = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self._SPECIAL_TOKENS,
            pretokenizer="bytelevel",
            initial_vocab=subword_vocab,
            initial_merges=subword_merges,
            max_superword_words=self.max_superword_words,
            forbid_colon_g=True,
            progress_callback=self._wandb_curve_cb(use_regex=False)
            if self.logger.enabled
            else None,
            progress_every=self.eval_every,
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
