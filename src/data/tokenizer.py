"""Tokenizer runtime loading. Training lives in src/data/tokenizer_trainer.py."""

import os

from tokenizers import Tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a trained tokenizer from disk."""
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
