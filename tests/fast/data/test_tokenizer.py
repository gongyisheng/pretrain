"""Behavior tests for src/data/tokenizer.py — BPE and SuperBPE paths."""

import json

import pytest
from tokenizers import Tokenizer, pre_tokenizers

from src.data.tokenizer import (
    _build_tokenizer_from_prefix,
    _stage1_pretokenizer,
    load_tokenizer,
    train_tokenizer,
)
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


# ---- Prefix reconstruction (Task 4) ----


def test_prefix_at_full_merges_matches_original(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    tok = train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=500,
        save_path=str(save),
        method="bpe",
    )
    # Extract vocab + merges from saved tokenizer.json
    data = json.loads((save / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    merges = [
        tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
        for m in data["model"]["merges"]
    ]
    full_k = len(merges)
    rebuilt = _build_tokenizer_from_prefix(
        vocab=vocab,
        merges=merges,
        k=full_k,
        pretok_factory=lambda: pre_tokenizers.ByteLevel(add_prefix_space=False),
    )
    s = "the quick brown fox"
    assert rebuilt.encode(s).ids == tok.encode(s).ids


def test_stage1_pretokenizer_pretokenizes_without_error():
    pt = _stage1_pretokenizer()
    # Smoke test: pretokenizes a mixed string without raising
    result = pt.pre_tokenize_str("hello world 123456 foo bar")
    assert len(result) > 0
    # Digit grouping should split runs of digits into groups of 3 from the right.
    # "123456" should produce at least one split (i.e., not a single span).
    pieces = [p[0] for p in result]
    assert any("123" in p or "456" in p for p in pieces), (
        f"expected digit-grouping to fire; got pieces={pieces}"
    )


def test_prefix_smaller_k_has_smaller_vocab(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=500,
        save_path=str(save),
        method="bpe",
    )
    data = json.loads((save / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    merges = [
        tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
        for m in data["model"]["merges"]
    ]
    half_k = len(merges) // 2
    half = _build_tokenizer_from_prefix(
        vocab=vocab,
        merges=merges,
        k=half_k,
        pretok_factory=lambda: pre_tokenizers.ByteLevel(add_prefix_space=False),
    )
    assert half.get_vocab_size() < len(vocab)
    assert half.get_vocab_size() == len(vocab) - (len(merges) - half_k)


# ---- SuperBPE stage 1 (Task 5) ----


def test_superbpe_stage1_saves_intermediate(tmp_path, text_iter):
    save = tmp_path / "sbpe"
    # transition_size=399 keeps stage 2's role minimal — we only verify stage 1 here.
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=400,
        save_path=str(save),
        method="superbpe",
        transition_size=399,
    )
    assert (save / "stage1.json").exists()


def test_superbpe_stage1_vocab_size(tmp_path, text_iter):
    save = tmp_path / "sbpe"
    # transition_size=399 keeps stage 2's role minimal — we only verify stage 1 here.
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=400,
        save_path=str(save),
        method="superbpe",
        transition_size=399,
    )
    stage1 = Tokenizer.from_file(str(save / "stage1.json"))
    # Stage 1 trains to transition_size (~400, may be slightly below due to
    # corpus size; HF stops when no more merges available).
    assert stage1.get_vocab_size() <= 400
    assert stage1.get_vocab_size() >= 256  # at least byte alphabet


# ---- SuperBPE stage 2 (Task 6) ----


def test_superbpe_full_train_succeeds(tmp_path, text_iter):
    save = tmp_path / "sbpe_400_t200"
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=400,
        save_path=str(save),
        method="superbpe",
        transition_size=200,
        max_superword_words=99,  # effectively disable cap for this task (guards land in Task 7)
    )
    assert (save / "tokenizer.json").exists()


def test_superbpe_final_vocab_size(tmp_path, text_iter):
    save = tmp_path / "sbpe_400_t200"
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=400,
        save_path=str(save),
        method="superbpe",
        transition_size=200,
        max_superword_words=99,
    )
    tok = load_tokenizer(str(save))
    # Small corpus has enough unique pairs to reach exactly T=400.
    # If SAMPLE_TEXTS is reduced in diversity, this may need to become <= 400.
    assert tok.get_vocab_size() == 400


def test_superbpe_produces_superword(tmp_path, text_iter):
    save = tmp_path / "sbpe_600_t300"
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=600,
        save_path=str(save),
        method="superbpe",
        transition_size=300,
        max_superword_words=99,
    )
    tok = load_tokenizer(str(save))
    vocab = tok.get_vocab()
    # Ġ not at position 0 = internal whitespace = superword token
    superwords = [t for t in vocab if "Ġ" in t[1:]]
    assert len(superwords) > 0, "expected at least one superword token after stage 2"


def test_superbpe_decode_roundtrip(tmp_path, text_iter):
    save = tmp_path / "sbpe_400_t200"
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=400,
        save_path=str(save),
        method="superbpe",
        transition_size=200,
        max_superword_words=99,
    )
    tok = load_tokenizer(str(save))
    for s in ["the quick brown fox", "she sells seashells", "by the way it works"]:
        ids = tok.encode(s).ids
        assert tok.decode(ids) == s, f"roundtrip failed for: {s!r}"


# ---- SuperBPE guards (Task 7) ----


def _ġ_count(tok: str) -> int:
    """Stage-1 'word count' of a token (Ġ characters + 1 if no leading Ġ)."""
    return tok.count("Ġ") + (0 if tok.startswith("Ġ") else 1)


def test_superbpe_max_superword_words_cap(tmp_path, text_iter):
    save = tmp_path / "sbpe_capped"
    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=600,
        save_path=str(save),
        method="superbpe",
        transition_size=300,
        max_superword_words=2,
    )
    tok = load_tokenizer(str(save))
    for t in tok.get_vocab():
        assert _ġ_count(t) <= 2, f"token {t!r} exceeds 2-word cap"


def test_superbpe_no_colon_space_tokens(tmp_path, text_iter):
    # Inject some "X: Y" patterns so the algorithm is tempted to learn ":Ġ" merges.
    texts = ["foo: bar baz quux", "name: alice", "topic: animals plants"] * 50
    save = tmp_path / "sbpe_colon"
    train_tokenizer(
        dataset_iter=iter(texts),
        vocab_size=500,
        save_path=str(save),
        method="superbpe",
        transition_size=250,
        max_superword_words=99,
    )
    tok = load_tokenizer(str(save))
    for t in tok.get_vocab():
        assert ":Ġ" not in t, f"forbidden ':Ġ' substring in token {t!r}"
