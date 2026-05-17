"""Tokenizer training (BPE and SuperBPE).

Two methods are supported:
- "bpe": standard byte-level BPE (HuggingFace tokenizers).
- "superbpe": two-stage curriculum from arXiv:2503.13423.

All trainers emit a HuggingFace-compatible tokenizer.json under save_path/.
Runtime loading lives in src/data/tokenizer.py.
"""

import heapq
import itertools
import json
import os
import time
from collections import Counter
from typing import Callable, Iterable

import numpy as np
from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm
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
            for partial, vocab_size in tqdm(
                _iter_partial_tokenizers(
                    tokenizer,
                    lambda: pre_tokenizers.ByteLevel(add_prefix_space=False),
                    self.eval_every,
                ),
                desc="[bpe-eval]",
                dynamic_ncols=True,
            ):
                self.logger.log(self.metrics.build_eval_log_dict(partial, vocab_size))
        return tokenizer

    # ---- SuperBPE ----

    def _train_superbpe(
        self, make_train_iter: Callable[[], Iterable[str]]
    ) -> Tokenizer:

        # Paper-default whitespace pretokenization regex (from HF tokenizers / GPT-2,
        # without the GPT-2 contraction-split rule). Used in stage 1 only.
        _WHITESPACE_REGEX = r" ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        # Digit-grouping pretok: split runs of digits into groups of 3 from the right.
        _DIGIT_REGEX = r"(?=(\d{3})+(?!\d))"

        def bpe_pretokenizer_factory():
            return pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(Regex(_WHITESPACE_REGEX), behavior="isolated"),
                    pre_tokenizers.Split(Regex(_DIGIT_REGEX), behavior="isolated"),
                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
                ]
            )

        # ---- Stage 1: HF BPE with whitespace + ByteLevel pretokenizer ----
        t0 = time.perf_counter()
        bpe_tokenizer = Tokenizer(models.BPE())
        bpe_tokenizer.pre_tokenizer = bpe_pretokenizer_factory()
        bpe_tokenizer.decoder = decoders.ByteLevel()
        bpe_trainer = trainers.BpeTrainer(
            vocab_size=self.transition_size,
            special_tokens=list(self._SPECIAL_TOKENS),
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        bpe_tokenizer.train_from_iterator(make_train_iter(), trainer=bpe_trainer)
        bpe_tokenizer.save(os.path.join(self.save_path, "bpe_tokenizer.json"))
        bpe_tokenizer_vocab_size = bpe_tokenizer.get_vocab_size()

        if self.logger.enabled:
            for partial, vocab_size in _iter_partial_tokenizers(
                bpe_tokenizer, bpe_pretokenizer_factory, self.eval_every
            ):
                self.logger.log(self.metrics.build_eval_log_dict(partial, vocab_size))

        print(
            f"SuperBPE stage 1 done: vocab_size={bpe_tokenizer_vocab_size} "
            f"({time.perf_counter() - t0:.2f}s)"
        )

        # ---- Stage 2 setup ----
        t0 = time.perf_counter()
        with open(os.path.join(self.save_path, "bpe_tokenizer.json")) as f:
            bpe_data = json.loads(f.read())
        bpe_vocab: dict = dict(bpe_data["model"]["vocab"])  # str -> int
        bpe_merges: list = [
            tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
            for m in bpe_data["model"]["merges"]
        ]

        # count num of whole words
        inverse_vocab_map: dict = {i: tok for tok, i in bpe_vocab.items()}
        word_count_of: dict = {
            i: tok.count("Ġ") + (0 if tok.startswith("Ġ") else 1)
            for i, tok in inverse_vocab_map.items()
        }

        # re-encode the corpus with stage 1
        n_docs = sum(1 for _ in make_train_iter())
        encode_batch_size = 1000
        docs: list[list[int]] = []
        stream = iter(make_train_iter())
        with tqdm(total=n_docs, desc="Tokenize words", dynamic_ncols=True) as pbar:
            while True:
                batch = list(itertools.islice(stream, encode_batch_size))
                if not batch:
                    break
                for enc in bpe_tokenizer.encode_batch(batch, add_special_tokens=False):
                    docs.append(enc.ids)
                pbar.update(len(batch))

        # 99th-percentile truncation: cap document length to mitigate duplication
        if len(docs) >= 100:
            lengths = np.array([len(d) for d in docs])
            p99 = int(np.percentile(lengths, 99))
            docs = [d[:p99] for d in docs]

        # Inverted-index state. Each doc is a token list with -1 marking
        # tombstoned slots. `positions[t]` tracks where token id t currently
        # lives so we can jump straight to those slots instead of scanning
        # every doc. Prev/next neighbors are recovered by scanning past
        # tombstones at lookup time (no extra storage).
        docs_tokens: list[list[int]] = [list(ids) for ids in docs]
        positions: dict[int, set[tuple[int, int]]] = {}
        pair_counts: Counter = Counter()
        for d, ids in enumerate(docs_tokens):
            for i, t in enumerate(ids):
                positions.setdefault(t, set()).add((d, i))
            for a, b in zip(ids[:-1], ids[1:]):
                pair_counts[(a, b)] += 1
        # Free the original `docs` list — docs_tokens owns the data now.
        docs = None  # type: ignore[assignment]

        # Lazy max-heap on (-count, -a, -b). Stale entries are skipped at pop
        # time by checking against pair_counts. Tuple ordering preserves the
        # original tiebreaker: largest count first, then lex-largest pair.
        heap: list = [(-cnt, -a, -b) for (a, b), cnt in pair_counts.items()]
        heapq.heapify(heap)

        vocab = dict(bpe_vocab)
        merges = list(bpe_merges)
        forbidden_substr = ":Ġ"
        n_accepted = 0
        n_blacklisted = 0

        pbar = tqdm(
            total=self.vocab_size,
            initial=len(vocab),
            desc="Compute merges",
            dynamic_ncols=True,
        )
        while len(vocab) < self.vocab_size:
            # Pop until a heap entry matches pair_counts (non-stale).
            a = b = None
            while heap:
                neg_cnt, neg_a, neg_b = heap[0]
                cand_a, cand_b = -neg_a, -neg_b
                cnt = -neg_cnt
                if cnt > 0 and pair_counts.get((cand_a, cand_b), 0) == cnt:
                    heapq.heappop(heap)
                    a, b = cand_a, cand_b
                    break
                heapq.heappop(heap)
            if a is None:
                break

            new_str = inverse_vocab_map[a] + inverse_vocab_map[b]
            new_word_count = word_count_of[a] + word_count_of[b]
            if new_word_count > self.max_superword_words or forbidden_substr in new_str:
                del pair_counts[(a, b)]
                n_blacklisted += 1
                continue
            new_id = len(vocab)
            vocab[new_str] = new_id
            inverse_vocab_map[new_id] = new_str
            word_count_of[new_id] = new_word_count
            merges.append((inverse_vocab_map[a], inverse_vocab_map[b]))
            _apply_merge_indexed(
                docs_tokens, positions, pair_counts, heap, a, b, new_id
            )
            n_accepted += 1
            pbar.update(1)

            # ---- In-loop stage-2 curve point ----
            cur_size = len(vocab)
            if self.logger.enabled and (
                cur_size % self.eval_every == 0 or cur_size == self.vocab_size
            ):
                tmp = Tokenizer(models.BPE(vocab=vocab, merges=merges))
                tmp.pre_tokenizer = pre_tokenizers.ByteLevel(
                    add_prefix_space=False, use_regex=False
                )
                tmp.decoder = decoders.ByteLevel()
                self.logger.log(self.metrics.build_train_log_dict(tmp, cur_size))
        pbar.close()

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


def _apply_merge_indexed(
    docs_tokens: list,
    positions: dict,
    pair_counts,  # Counter
    heap: list,
    a: int,
    b: int,
    new_id: int,
) -> None:
    """Apply merge (a, b) -> new_id using the inverted index.

    Visits only the slots where `a` lives (via `positions[a]`), not every doc.
    Neighbors (the b slot, the slots either side) are recovered by scanning
    past tombstones (-1) in `docs_tokens[d]`. pair_counts and the lazy heap
    are updated to reflect the broken/new neighbor pairs.

    Positions are processed in (doc, slot) order so overlapping cases like
    `a a a` consume left-to-right, matching the legacy algorithm.
    """
    # Snapshot in deterministic order; positions[a] mutates during the loop.
    targets = sorted(positions.get(a, ()))
    pos_a = positions[a]
    pos_b = positions.get(b)
    pos_new = positions.setdefault(new_id, set())

    for d, p in targets:
        # Stale entry — `a` was already consumed by an earlier merge in this pass.
        if (d, p) not in pos_a or docs_tokens[d][p] != a:
            continue
        tok_arr = docs_tokens[d]
        n = len(tok_arr)

        # Scan rightward past tombstones to find b.
        q = p + 1
        while q < n and tok_arr[q] == -1:
            q += 1
        if q >= n or tok_arr[q] != b:
            continue

        # Scan leftward past tombstones for the previous live token.
        r = p - 1
        while r >= 0 and tok_arr[r] == -1:
            r -= 1

        # Scan rightward past q for the next live token after b.
        s = q + 1
        while s < n and tok_arr[s] == -1:
            s += 1
        if s >= n:
            s = -1

        # Tombstone q (the b slot).
        tok_arr[q] = -1
        pos_b.discard((d, q))

        # Rewrite p (was a) to new_id.
        tok_arr[p] = new_id
        pos_a.discard((d, p))
        pos_new.add((d, p))

        # Decrement (a, b) once per occurrence consumed.
        pair_counts[(a, b)] -= 1

        # Left neighbor: (prev_tok, a) -> (prev_tok, new_id).
        if r >= 0:
            prev_tok = tok_arr[r]
            pair_counts[(prev_tok, a)] -= 1
            if pair_counts[(prev_tok, a)] <= 0:
                del pair_counts[(prev_tok, a)]
            else:
                heapq.heappush(heap, (-pair_counts[(prev_tok, a)], -prev_tok, -a))
            pair_counts[(prev_tok, new_id)] += 1
            heapq.heappush(heap, (-pair_counts[(prev_tok, new_id)], -prev_tok, -new_id))

        # Right neighbor: (b, next_tok) -> (new_id, next_tok).
        if s != -1:
            next_tok = tok_arr[s]
            pair_counts[(b, next_tok)] -= 1
            if pair_counts[(b, next_tok)] <= 0:
                del pair_counts[(b, next_tok)]
            else:
                heapq.heappush(heap, (-pair_counts[(b, next_tok)], -b, -next_tok))
            pair_counts[(new_id, next_tok)] += 1
            heapq.heappush(heap, (-pair_counts[(new_id, next_tok)], -new_id, -next_tok))

    # The merged pair is exhausted (count reached 0 along the way, or no
    # adjacencies were ever present); drop the entry if it's still around.
    if (a, b) in pair_counts and pair_counts[(a, b)] <= 0:
        del pair_counts[(a, b)]


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
