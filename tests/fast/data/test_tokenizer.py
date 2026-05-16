"""Tests for src/data/tokenizer.py — load_tokenizer."""

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.data.tokenizer import load_tokenizer


def _save_minimal_tokenizer(save_dir, vocab_size: int = 300) -> Tokenizer:
    """Train and save a tiny byte-level BPE tokenizer to `save_dir/tokenizer.json`."""
    save_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=False,
    )
    corpus = ["the quick brown fox jumps over the lazy dog"] * 50
    tok.train_from_iterator(iter(corpus), trainer=trainer)
    tok.save(str(save_dir / "tokenizer.json"))
    return tok


def test_load_tokenizer_returns_tokenizer(tmp_path):
    save_dir = tmp_path / "tok"
    _save_minimal_tokenizer(save_dir)
    loaded = load_tokenizer(str(save_dir))
    assert isinstance(loaded, Tokenizer)
    assert loaded.get_vocab_size() > 0


def test_load_tokenizer_matches_saved(tmp_path):
    save_dir = tmp_path / "tok"
    saved = _save_minimal_tokenizer(save_dir)
    loaded = load_tokenizer(str(save_dir))
    for s in ["the quick brown fox", "lazy dog", "the"]:
        assert loaded.encode(s).ids == saved.encode(s).ids


def test_load_tokenizer_decode_roundtrip(tmp_path):
    save_dir = tmp_path / "tok"
    _save_minimal_tokenizer(save_dir)
    loaded = load_tokenizer(str(save_dir))
    for s in ["the quick brown fox", "the lazy dog"]:
        assert loaded.decode(loaded.encode(s).ids) == s


def test_load_tokenizer_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        load_tokenizer(str(tmp_path / "does_not_exist"))
