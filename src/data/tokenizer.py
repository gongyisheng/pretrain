"""Tokenizer training and loading.

Two methods are supported:
- "bpe": standard byte-level BPE (HuggingFace tokenizers).
- "superbpe": two-stage curriculum from arXiv:2503.13423 (see _train_superbpe_tokenizer).

All trainers emit a HuggingFace-compatible tokenizer.json under save_path/.
"""

import json
import os
import time
from collections import Counter
from typing import Iterable, Optional, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_tokenizer(
    dataset_iter: Iterable[str],
    vocab_size: int = 50257,
    save_path: str = "tokenizers/custom",
    method: str = "bpe",
    transition_size: Optional[int] = None,
    max_superword_words: int = 4,
    special_tokens: Sequence[str] = ("<|endoftext|>",),
    wandb_enabled: bool = False,
    wandb_project: str = "superbpe",
    wandb_eval_every: int = 5000,
    eval_num_docs: int = 1000,
) -> Tokenizer:
    """Train a tokenizer and save it under save_path/tokenizer.json.

    Args:
        dataset_iter: iterable of text strings (consumed in order, no shuffling).
        vocab_size: final vocabulary size.
        save_path: directory to save tokenizer.json (and other artifacts).
        method: "bpe" or "superbpe".
        transition_size: required when method="superbpe"; stage 1 trains BPE with
            whitespace pretokenization up to this vocab size, stage 2 continues
            without whitespace pretokenization up to vocab_size.
        max_superword_words: when method="superbpe", cap on superword length
            measured in stage-1 word units. Ignored when method="bpe".
        special_tokens: tokens to add with reserved IDs at the start of the vocab.
        wandb_enabled: if True, log an efficiency curve (vocab_size vs bytes/token)
            to W&B. Only used when method="superbpe".
        wandb_project: W&B project name for efficiency curve logging.
        wandb_eval_every: interval (in number of merges) at which to log curve
            points. Stage-1 uses merge-prefix count; stage-2 uses accepted merges.
        eval_num_docs: number of documents to hold out for efficiency evaluation.

    Returns:
        The trained `tokenizers.Tokenizer` instance.
    """
    if method == "bpe":
        return _train_bpe_tokenizer(dataset_iter, vocab_size, save_path, special_tokens)
    if method == "superbpe":
        if transition_size is None or not (0 < transition_size < vocab_size):
            raise ValueError(
                f"method='superbpe' requires 0 < transition_size < vocab_size; "
                f"got transition_size={transition_size}, vocab_size={vocab_size}"
            )
        return _train_superbpe_tokenizer(
            dataset_iter,
            vocab_size,
            transition_size,
            save_path,
            max_superword_words,
            special_tokens,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_eval_every=wandb_eval_every,
            eval_num_docs=eval_num_docs,
        )
    raise ValueError(f"unknown method: {method!r}; expected 'bpe' or 'superbpe'")


def load_tokenizer(path: str) -> Tokenizer:
    """Load a trained tokenizer from disk."""
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json"))


# ---- BPE (existing behavior, now private) ----


def _train_bpe_tokenizer(
    dataset_iter: Iterable[str],
    vocab_size: int,
    save_path: str,
    special_tokens: Sequence[str],
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        show_progress=True,
    )
    tokenizer.train_from_iterator(dataset_iter, trainer=trainer)

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    print(
        f"BPE tokenizer saved to {save_path}/ (vocab_size={tokenizer.get_vocab_size()})"
    )
    return tokenizer


# ---- Helpers shared by superbpe + W&B curve logging ----

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


# ---- SuperBPE (added in later tasks) ----


def _train_superbpe_tokenizer(
    dataset_iter: Iterable[str],
    vocab_size: int,
    transition_size: int,
    save_path: str,
    max_superword_words: int,
    special_tokens: Sequence[str],
    *,
    wandb_enabled: bool = False,
    wandb_project: str = "superbpe",
    wandb_eval_every: int = 5000,
    eval_num_docs: int = 1000,
) -> Tokenizer:
    os.makedirs(save_path, exist_ok=True)
    # We must materialize the iterator into a list — stage 1 and stage 2 each
    # consume it once, and the held-out / re-encoding paths in later tasks need
    # multiple passes. For 100 MB at ~3B/char × N docs this is fine on the
    # workstation. The caller is responsible for sizing the input.
    texts = list(dataset_iter)
    if not texts:
        raise ValueError("dataset_iter produced no text")
    n_docs = len(texts)

    # ---- Held-out split for W&B efficiency evaluation ----
    if eval_num_docs > 0 and eval_num_docs < len(texts):
        held_out = texts[:eval_num_docs]
        texts = texts[eval_num_docs:]
    else:
        held_out = texts[: max(1, min(eval_num_docs, len(texts)))]

    # ---- W&B init ----
    if wandb_enabled:
        import wandb

        wandb.init(
            project=wandb_project,
            name=f"superbpe_T{vocab_size}_t{transition_size}",
            config={
                "T": vocab_size,
                "t": transition_size,
                "method": "superbpe",
                "max_superword_words": max_superword_words,
                "eval_num_docs": len(held_out),
            },
        )
        wandb.define_metric("vocab_size")
        wandb.define_metric("bytes_per_token", step_metric="vocab_size")

    # ---- Curve logging helper (closes over wandb_enabled and held_out) ----
    def _log_point(tok: "Tokenizer", v: int) -> None:
        if not wandb_enabled:
            return
        import wandb
        from src.eval.tokenizer import _bytes_per_token

        bpt = _bytes_per_token(tok, held_out)
        wandb.log({"vocab_size": v, "bytes_per_token": bpt})

    # ---- Stage 1: HF BPE with whitespace + ByteLevel pretokenizer ----
    t0 = time.perf_counter()
    stage1 = Tokenizer(models.BPE())
    stage1.pre_tokenizer = _stage1_pretokenizer()
    stage1.decoder = decoders.ByteLevel()
    stage1_trainer = trainers.BpeTrainer(
        vocab_size=transition_size,
        special_tokens=list(special_tokens),
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    stage1.train_from_iterator(iter(texts), trainer=stage1_trainer)
    stage1.save(os.path.join(save_path, "stage1.json"))
    stage1_seconds = time.perf_counter() - t0
    print(
        f"SuperBPE stage 1 done: vocab_size={stage1.get_vocab_size()} ({stage1_seconds:.2f}s)"
    )

    # ---- Stage 2 setup ----
    # (stage1.json is read here; post-hoc stage-1 curve points logged below)
    t0 = time.perf_counter()
    with open(os.path.join(save_path, "stage1.json")) as f:
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

    # Cache stage-1 vocab size — used in curve points, in-loop logging,
    # the meta file, and the safety final-point guard.
    stage1_size = stage1.get_vocab_size()

    # ---- Post-hoc stage-1 curve points ----
    if wandb_enabled:
        base_count = len(s1_vocab) - len(s1_merges)  # alphabet + specials
        for k in range(wandb_eval_every, len(s1_merges) + 1, wandb_eval_every):
            partial = _build_tokenizer_from_prefix(
                s1_vocab,
                s1_merges,
                k,
                pretok_factory=_stage1_pretokenizer,
            )
            _log_point(partial, base_count + k)
        # Forced point at exactly stage-1 final vocab.
        _log_point(stage1, stage1_size)

    # Re-encode the entire corpus with stage 1.
    docs: list[list[int]] = [
        stage1.encode(t, add_special_tokens=False).ids for t in texts
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

    while len(vocab) < vocab_size and pair_counts:
        # Explicit tiebreaker — see spec Reproducibility section.
        (a, b), best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count <= 0:
            break
        new_str = inv_vocab[a] + inv_vocab[b]
        new_word_count = word_count_of[a] + word_count_of[b]
        # Guards.
        if new_word_count > max_superword_words or forbidden_substr in new_str:
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
        if wandb_enabled and (
            (cur_size - stage1_size) % wandb_eval_every == 0 or cur_size == vocab_size
        ):
            tmp = Tokenizer(models.BPE(vocab=vocab, merges=merges))
            tmp.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False
            )
            tmp.decoder = decoders.ByteLevel()
            _log_point(tmp, cur_size)

    stage2_seconds = time.perf_counter() - t0
    print(
        f"SuperBPE stage 2 done: vocab_size={len(vocab)} "
        f"(stage-2 merges accepted: {n_accepted}, blacklisted: {n_blacklisted}, "
        f"{stage2_seconds:.2f}s)"
    )

    # ---- Final assembly ----
    final = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    final.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    final.decoder = decoders.ByteLevel()
    final.save(os.path.join(save_path, "tokenizer.json"))

    # ---- training_meta.json ----
    meta = {
        "T": vocab_size,
        "t": transition_size,
        "method": "superbpe",
        "max_superword_words": max_superword_words,
        "n_docs": n_docs,
        "stage1_seconds": stage1_seconds,
        "stage2_seconds": stage2_seconds,
        "stage1_vocab_size": stage1_size,
        "final_vocab_size": len(vocab),
        "n_stage2_merges_accepted": n_accepted,
        "n_stage2_merges_blacklisted": n_blacklisted,
    }
    with open(os.path.join(save_path, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"SuperBPE tokenizer saved to {save_path}/")

    # ---- W&B finalize ----
    if wandb_enabled:
        # Final point only if the loop exited early (pair_counts exhausted).
        # When the loop runs to completion (len(vocab) == vocab_size), the
        # in-loop `or cur_size == vocab_size` branch already logged that point.
        if len(vocab) != vocab_size:
            _log_point(final, len(vocab))
        import wandb

        wandb.finish()

    return final
