"""Tokenizer training and loading.

Two methods are supported:
- "bpe": standard byte-level BPE (HuggingFace tokenizers).
- "superbpe": two-stage curriculum from arXiv:2503.13423 (see _train_superbpe_tokenizer).

All trainers emit a HuggingFace-compatible tokenizer.json under save_path/.
"""

import os
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


# ---- SuperBPE (added in later tasks) ----


def _train_superbpe_tokenizer(
    dataset_iter: Iterable[str],
    vocab_size: int,
    transition_size: int,
    save_path: str,
    max_superword_words: int,
    special_tokens: Sequence[str],
) -> Tokenizer:
    raise NotImplementedError("SuperBPE trainer lands in Task 8")
