"""Tokenizer training (BPE and SuperBPE).

Two methods are supported:
- "bpe": standard byte-level BPE (HuggingFace tokenizers).
- "superbpe": two-stage curriculum from arXiv:2503.13423.

All trainers emit a HuggingFace-compatible tokenizer.json under save_path/.
Runtime loading lives in src/data/tokenizer.py.
"""

import itertools
import json
import os
import time
from collections import Counter
from typing import Callable, Iterable

import wandb
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

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

    Method-specific kwargs live in `config.data.tokenizer_train_kwargs` and are
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
        self.train_kwargs = dict(config.data.tokenizer_train_kwargs)
        self.eval_every = config.data.tokenizer_train_eval_every
        self.eval_num_docs = self.train_kwargs.get("eval_num_docs", 1000)

        if self.train_method not in ("bpe", "superbpe"):
            raise ValueError(
                f"unknown method: {self.train_method!r}; expected 'bpe' or 'superbpe'"
            )

        # SuperBPE-specific knobs, validated up front so `_train_superbpe` can
        # assume they're well-formed.
        self.transition_size: int | None = None
        self.max_superword_words: int = 4
        if self.train_method == "superbpe":
            ts = self.train_kwargs.get("transition_size")
            if ts is None or not (0 < ts <= self.vocab_size):
                raise ValueError(
                    f"method='superbpe' requires 0 < transition_size <= vocab_size; "
                    f"got transition_size={ts}, vocab_size={self.vocab_size}"
                )
            self.transition_size = ts
            self.max_superword_words = self.train_kwargs.get("max_superword_words", 4)

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
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self._SPECIAL_TOKENS),
            show_progress=True,
        )
        tokenizer.train_from_iterator(make_train_iter(), trainer=trainer)

        if self.logger.enabled:
            for partial, vocab_size in _iter_partial_tokenizers(
                tokenizer,
                lambda: pre_tokenizers.ByteLevel(add_prefix_space=False),
                self.eval_every,
            ):
                self.logger.log(self.metrics.build_eval_log_dict(partial, vocab_size))
        return tokenizer

    # ---- SuperBPE ----

    def _train_superbpe(
        self, make_train_iter: Callable[[], Iterable[str]]
    ) -> Tokenizer:
        # ---- Stage 1: HF BPE with whitespace + ByteLevel pretokenizer ----
        t0 = time.perf_counter()
        stage1 = Tokenizer(models.BPE())
        stage1.pre_tokenizer = _stage1_pretokenizer()
        stage1.decoder = decoders.ByteLevel()
        stage1_trainer = trainers.BpeTrainer(
            vocab_size=self.transition_size,
            special_tokens=list(self._SPECIAL_TOKENS),
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        stage1.train_from_iterator(make_train_iter(), trainer=stage1_trainer)
        stage1.save(os.path.join(self.save_path, "stage1.json"))
        stage1_size = stage1.get_vocab_size()
        print(
            f"SuperBPE stage 1 done: vocab_size={stage1_size} "
            f"({time.perf_counter() - t0:.2f}s)"
        )

        # ---- Stage 2 setup ----
        t0 = time.perf_counter()
        with open(os.path.join(self.save_path, "stage1.json")) as f:
            stage1_data = json.loads(f.read())
        s1_vocab: dict = dict(stage1_data["model"]["vocab"])  # str -> int
        s1_merges: list = [
            tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
            for m in stage1_data["model"]["merges"]
        ]
        inv_vocab: dict = {i: tok for tok, i in s1_vocab.items()}
        # Per-stage-1-token word count: |Ġ| + 1 if no leading Ġ, else |Ġ|.
        word_count_of: dict = {
            i: tok.count("Ġ") + (0 if tok.startswith("Ġ") else 1)
            for i, tok in inv_vocab.items()
        }

        # ---- Post-hoc stage-1 curve points ----
        if self.logger.enabled:
            for partial, vocab_size in _iter_partial_tokenizers(
                stage1, _stage1_pretokenizer, self.eval_every
            ):
                self.logger.log(self.metrics.build_eval_log_dict(partial, vocab_size))

        # Re-encode the corpus with stage 1 (fresh stream — no raw-text materialization).
        docs: list[list[int]] = [
            stage1.encode(t, add_special_tokens=False).ids for t in make_train_iter()
        ]

        # 99th-percentile truncation: cap document length to mitigate duplication
        # artifacts from very long outliers (paper section 3).
        if len(docs) >= 100:
            import numpy as np

            lengths = np.array([len(d) for d in docs])
            p99 = int(np.percentile(lengths, 99))
            docs = [d[:p99] for d in docs]

        # Pair counts across each document, no whitespace boundary.
        pair_counts: Counter = Counter()
        for ids in docs:
            for a, b in zip(ids[:-1], ids[1:]):
                pair_counts[(a, b)] += 1

        vocab = dict(s1_vocab)
        merges = list(s1_merges)
        forbidden_substr = ":Ġ"
        n_accepted = 0
        n_blacklisted = 0

        while len(vocab) < self.vocab_size and pair_counts:
            # Explicit tiebreaker — see spec Reproducibility section.
            (a, b), best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if best_count <= 0:
                break
            new_str = inv_vocab[a] + inv_vocab[b]
            new_word_count = word_count_of[a] + word_count_of[b]
            # Guards.
            if new_word_count > self.max_superword_words or forbidden_substr in new_str:
                del pair_counts[(a, b)]
                n_blacklisted += 1
                continue
            new_id = len(vocab)
            vocab[new_str] = new_id
            inv_vocab[new_id] = new_str
            word_count_of[new_id] = new_word_count
            merges.append((inv_vocab[a], inv_vocab[b]))
            _apply_merge_inplace(docs, pair_counts, a, b, new_id)
            n_accepted += 1

            # ---- In-loop stage-2 curve point ----
            cur_size = len(vocab)
            if self.logger.enabled and (
                (cur_size - stage1_size) % self.eval_every == 0
                or cur_size == self.vocab_size
            ):
                tmp = Tokenizer(models.BPE(vocab=vocab, merges=merges))
                tmp.pre_tokenizer = pre_tokenizers.ByteLevel(
                    add_prefix_space=False, use_regex=False
                )
                tmp.decoder = decoders.ByteLevel()
                self.logger.log(self.metrics.build_train_log_dict(tmp, cur_size))

        print(
            f"SuperBPE stage 2 done: vocab_size={len(vocab)} "
            f"(stage-2 merges accepted: {n_accepted}, blacklisted: {n_blacklisted}, "
            f"{time.perf_counter() - t0:.2f}s)"
        )

        # ---- Final assembly ----
        final = Tokenizer(models.BPE(vocab=vocab, merges=merges))
        final.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False
        )
        final.decoder = decoders.ByteLevel()

        # Final point only if the loop exited early (pair_counts exhausted).
        # When the loop runs to completion (len(vocab) == vocab_size), the
        # in-loop branch already logged that point.
        if self.logger.enabled and len(vocab) != self.vocab_size:
            self.logger.log(self.metrics.build_eval_log_dict(final, len(vocab)))

        return final


def _iter_partial_tokenizers(tokenizer: Tokenizer, pretok_factory, eval_every: int):
    """Yield (partial_tokenizer, vocab_size) for every `eval_every` merges, plus
    a final entry for the full trained tokenizer.
    """
    data = json.loads(tokenizer.to_str())
    vocab = data["model"]["vocab"]
    merges = [
        tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
        for m in data["model"]["merges"]
    ]
    base_count = len(vocab) - len(merges)  # alphabet + specials
    for k in range(eval_every, len(merges) + 1, eval_every):
        yield (
            _build_tokenizer_from_prefix(vocab, merges, k, pretok_factory),
            base_count + k,
        )
    yield tokenizer, tokenizer.get_vocab_size()


# ---- Helpers shared by superbpe + curve reconstruction ----

# Paper-default whitespace pretokenization regex (from HF tokenizers / GPT-2,
# without the GPT-2 contraction-split rule). Used in stage 1 only.
_WHITESPACE_REGEX = r" ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
# Digit-grouping pretok: split runs of digits into groups of 3 from the right.
_DIGIT_REGEX = r"(?=(\d{3})+(?!\d))"


def _stage1_pretokenizer():
    """Build the stage-1 pretokenizer: regex Split + digit Split + ByteLevel."""
    from tokenizers import Regex

    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(_WHITESPACE_REGEX), behavior="isolated"),
            pre_tokenizers.Split(Regex(_DIGIT_REGEX), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )


def _apply_merge_inplace(
    docs: list,
    pair_counts,  # Counter
    a: int,
    b: int,
    new_id: int,
) -> None:
    """Replace every (a, b) adjacency in docs with new_id; update pair_counts.

    Standard incremental BPE update. We rebuild each doc with a left-to-right
    scan, and for every replaced position we decrement the broken-up neighbor
    pairs and increment the newly-formed ones.
    """
    # Drop the merged pair itself from pair_counts; it's about to be consumed.
    del pair_counts[(a, b)]
    for doc_idx, ids in enumerate(docs):
        if a not in ids:
            continue
        new_ids: list[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                # Decrement broken-up neighbor pairs.
                if new_ids:
                    prev = new_ids[-1]
                    pair_counts[(prev, a)] -= 1
                    if pair_counts[(prev, a)] <= 0:
                        del pair_counts[(prev, a)]
                if i + 2 < n:
                    nxt = ids[i + 2]
                    pair_counts[(b, nxt)] -= 1
                    if pair_counts[(b, nxt)] <= 0:
                        del pair_counts[(b, nxt)]
                # Append the merged token.
                new_ids.append(new_id)
                # Increment newly-formed neighbor pairs.
                if len(new_ids) >= 2:
                    pair_counts[(new_ids[-2], new_id)] += 1
                if i + 2 < n:
                    pair_counts[(new_id, ids[i + 2])] += 1
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        docs[doc_idx] = new_ids


def _build_tokenizer_from_prefix(
    vocab: dict,
    merges: list,
    k: int,
    pretok_factory,
) -> Tokenizer:
    """Reconstruct a Tokenizer from the first `k` merges.

    BPE at vocab_size = |alphabet| + |specials| + k is fully determined by
    `(alphabet ∪ specials ∪ k-merge-derived-tokens, first k merges)`.

    Args:
        vocab: full vocab dict (token_str -> id) from a trained tokenizer.
        merges: ordered merge list (each item a (left, right) tuple of strs).
        k: number of merges to include (0 <= k <= len(merges)).
        pretok_factory: zero-arg callable returning a pre_tokenizer.

    Returns:
        A `Tokenizer` whose merge list is `merges[:k]` and whose vocab is
        the subset of `vocab` consistent with those merges.
    """
    assert 0 <= k <= len(merges), f"k={k} out of range [0, {len(merges)}]"
    kept_merges = merges[:k]
    all_merge_results = {a + b for (a, b) in merges}
    # Tokens that are *not* merge results across the full merge list = alphabet + specials.
    base_tokens = {t for t in vocab if t not in all_merge_results}
    # Tokens produced by the first k merges.
    merge_results = {a + b for (a, b) in kept_merges}
    # Any token referenced as a merge component must also be present.
    component_tokens = {part for (a, b) in kept_merges for part in (a, b)}
    kept_tokens = base_tokens | merge_results | (component_tokens & vocab.keys())
    # Preserve original IDs for kept tokens, compact IDs are reassigned by BPE
    # constructor based on insertion order, so we sort by original ID.
    for tok in kept_tokens:
        assert tok in vocab, f"kept token {tok!r} not in vocab — invariant violated"
    kept_pairs = sorted(
        ((tok, vocab[tok]) for tok in kept_tokens),
        key=lambda p: p[1],
    )
    new_vocab = {tok: i for i, (tok, _) in enumerate(kept_pairs)}

    tok = Tokenizer(models.BPE(vocab=new_vocab, merges=kept_merges))
    tok.pre_tokenizer = pretok_factory()
    tok.decoder = decoders.ByteLevel()
    return tok
