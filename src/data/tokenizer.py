"""Tokenizer training and loading.

Two methods are supported:
- "bpe": standard byte-level BPE (HuggingFace tokenizers).
- "superbpe": two-stage curriculum from arXiv:2503.13423 (see _train_superbpe_tokenizer).

All trainers emit a HuggingFace-compatible tokenizer.json under save_path/.
"""

import json
import os
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
) -> Tokenizer:
    os.makedirs(save_path, exist_ok=True)
    # We must materialize the iterator into a list — stage 1 and stage 2 each
    # consume it once, and the held-out / re-encoding paths in later tasks need
    # multiple passes. For 100 MB at ~3B/char × N docs this is fine on the
    # workstation. The caller is responsible for sizing the input.
    texts = list(dataset_iter)
    if not texts:
        raise ValueError("dataset_iter produced no text")

    # ---- Stage 1: HF BPE with whitespace + ByteLevel pretokenizer ----
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
    print(f"SuperBPE stage 1 done: vocab_size={stage1.get_vocab_size()}")

    # ---- Stage 2 setup ----
    with open(os.path.join(save_path, "stage1.json")) as f:
        stage1_data = json.loads(f.read())
    s1_vocab: dict = dict(stage1_data["model"]["vocab"])  # str -> int
    s1_merges: list = [
        tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
        for m in stage1_data["model"]["merges"]
    ]
    inv_vocab: dict = {i: tok for tok, i in s1_vocab.items()}

    # Re-encode the entire corpus with stage 1.
    docs: list[list[int]] = [
        stage1.encode(t, add_special_tokens=False).ids for t in texts
    ]

    # Pair counts across each document, no whitespace boundary.
    pair_counts: Counter = Counter()
    for ids in docs:
        for a, b in zip(ids[:-1], ids[1:]):
            pair_counts[(a, b)] += 1

    vocab = dict(s1_vocab)
    merges = list(s1_merges)

    # Stage 2 merge loop — no constraints (Task 7 will add them).
    while len(vocab) < vocab_size and pair_counts:
        (a, b), best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count <= 0:
            break
        new_str = inv_vocab[a] + inv_vocab[b]
        new_id = len(vocab)
        vocab[new_str] = new_id
        inv_vocab[new_id] = new_str
        merges.append((inv_vocab[a], inv_vocab[b]))
        _apply_merge_inplace(docs, pair_counts, a, b, new_id)

    print(
        f"SuperBPE stage 2 done: vocab_size={len(vocab)} "
        f"(stage-2 merges: {len(merges) - len(s1_merges)})"
    )

    # ---- Final assembly ----
    final = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    final.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    final.decoder = decoders.ByteLevel()
    final.save(os.path.join(save_path, "tokenizer.json"))
    print(f"SuperBPE tokenizer saved to {save_path}/")
    return final
