"""Tests for src/data/bpe.py — pure-Python BPE trainer."""

from src.data.bpe import BpeTrainer


def test_bpe_module_exports_trainer():
    assert BpeTrainer is not None
    assert callable(BpeTrainer)
