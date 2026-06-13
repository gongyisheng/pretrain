"""Tokenizer runtime loading. Training lives in src/training/trainer.py."""

import os

from tokenizers import Tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a trained tokenizer from disk."""
    tokenizer_file = os.path.join(path, "tokenizer.json")
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(
            f"tokenizer not found at {tokenizer_file}; "
            f"train one with scripts/train_tokenizer.py first"
        )
    return Tokenizer.from_file(tokenizer_file)
