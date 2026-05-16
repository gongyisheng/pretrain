"""Behavior tests for src/data/tokenizer.py — BPE and SuperBPE paths."""

import pytest

from src.data.tokenizer import load_tokenizer, train_tokenizer
from src.eval.tokenizer import _bytes_per_token, evaluate


# ---- Shared fixtures ----

SAMPLE_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "über naïve cafés au lait sont délicieux",
    "naïve über alles résumé façade",
] * 20  # 140 docs total; includes non-ASCII so byte tokens for ü/ï/é etc. are learned


@pytest.fixture
def text_iter():
    def _make():
        return iter(SAMPLE_TEXTS)

    return _make


# ---- BPE path (Task 2) ----


def test_bpe_train_produces_tokenizer_json(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=500,
        save_path=str(save),
        method="bpe",
    )
    assert (save / "tokenizer.json").exists()


def test_bpe_load_roundtrip(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    tok_train = train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=500,
        save_path=str(save),
        method="bpe",
    )
    tok_load = load_tokenizer(str(save))
    s = "the quick brown fox"
    assert tok_train.encode(s).ids == tok_load.encode(s).ids


def test_bpe_decode_roundtrip(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    tok = train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=500,
        save_path=str(save),
        method="bpe",
    )
    for s in ["hello world", "the quick brown fox", "über naïve cafés"]:
        ids = tok.encode(s).ids
        assert tok.decode(ids) == s


def test_unknown_method_raises(tmp_path, text_iter):
    with pytest.raises(ValueError, match="unknown method"):
        train_tokenizer(
            dataset_iter=text_iter(),
            vocab_size=500,
            save_path=str(tmp_path / "x"),
            method="not_a_real_method",
        )


@pytest.mark.parametrize("ts", [None, 0, -1, 500, 600])
def test_superbpe_invalid_transition_size_raises(tmp_path, text_iter, ts):
    with pytest.raises(ValueError):
        train_tokenizer(
            dataset_iter=text_iter(),
            vocab_size=500,
            save_path=str(tmp_path / "x"),
            method="superbpe",
            transition_size=ts,
        )


# ---- Eval (Task 3) ----


def test_bytes_per_token_sensible(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    tok = train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=500,
        save_path=str(save),
        method="bpe",
    )
    bpt = _bytes_per_token(tok, SAMPLE_TEXTS[:20])
    assert bpt > 1.0, f"expected >1 byte per token, got {bpt}"
    assert bpt < 20.0, f"absurdly high bytes/token: {bpt}"


def test_evaluate_returns_expected_keys(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=500,
        save_path=str(save),
        method="bpe",
    )
    result = evaluate(str(save), iter(SAMPLE_TEXTS[:20]))
    assert set(result.keys()) >= {
        "n_docs",
        "n_bytes",
        "n_tokens",
        "bytes_per_token",
        "tokens_per_byte",
    }
    assert result["n_docs"] == 20
    assert result["n_tokens"] > 0
    assert abs(result["bytes_per_token"] * result["tokens_per_byte"] - 1.0) < 1e-9
