"""Behavior tests for src/data/tokenizer_trainer.py — BPE and SuperBPE paths."""

import pytest
from tokenizers import Tokenizer

from src.data.tokenizer import load_tokenizer
from src.data.tokenizer_trainer import (
    TokenizerTrainer,
    encode_corpus_to_chunks,
)
from src.eval.tokenizer import _bytes_per_token, evaluate
from src.utils.config import DataConfig, LoggingConfig, ModelConfig, TrainConfig


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


def _trainer(
    save_path,
    vocab_size: int,
    method: str,
    *,
    eval_every: int = 5000,
    wandb_enabled: bool = False,
    logging_kwargs: dict | None = None,
    **method_kwargs,
) -> TokenizerTrainer:
    """Build a TokenizerTrainer with a minimal TrainConfig."""
    config = TrainConfig(
        model=ModelConfig(vocab_size=vocab_size),
        data=DataConfig(
            tokenizer_path=str(save_path),
            tokenizer_train_method=method,
            tokenizer_train_method_kwargs=method_kwargs,
            tokenizer_train_eval_every=eval_every,
        ),
        logging=LoggingConfig(**(logging_kwargs or {})),
    )
    return TokenizerTrainer(config, wandb_enabled=wandb_enabled)


# ---- BPE ----


def test_bpe_train_produces_tokenizer_json(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    _trainer(save, 500, "bpe").train(text_iter)
    assert (save / "tokenizer.json").exists()


def test_bpe_load_roundtrip(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    tok_train = _trainer(save, 500, "bpe").train(text_iter)
    tok_load = load_tokenizer(str(save))
    s = "the quick brown fox"
    assert tok_train.encode(s).ids == tok_load.encode(s).ids


def test_bpe_decode_roundtrip(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    tok = _trainer(save, 500, "bpe").train(text_iter)
    for s in ["hello world", "the quick brown fox", "über naïve cafés"]:
        ids = tok.encode(s).ids
        assert tok.decode(ids) == s


def test_unknown_method_raises(tmp_path, text_iter):
    with pytest.raises(ValueError, match="unknown method"):
        _trainer(tmp_path / "x", 500, "not_a_real_method").train(text_iter)


def test_bpe_vocab_size_respected(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    tok = _trainer(save, 500, "bpe").train(text_iter)
    # Trained vocab may equal the target (when corpus has enough merges) or
    # fall slightly below (when corpus is small). Asserting <= 500 catches an
    # overshoot regression while tolerating natural undershoots.
    assert tok.get_vocab_size() <= 500
    assert tok.get_vocab_size() >= 1  # sanity: non-empty vocab


@pytest.mark.parametrize("ts", [None, 0, -1, 600])
def test_superbpe_invalid_transition_size_raises(tmp_path, text_iter, ts):
    kwargs = {"transition_size": ts} if ts is not None else {}
    with pytest.raises(ValueError):
        _trainer(tmp_path / "x", 500, "superbpe", **kwargs).train(text_iter)


# ---- Eval ----


def test_bytes_per_token_sensible(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    tok = _trainer(save, 500, "bpe").train(text_iter)
    bpt = _bytes_per_token(tok, SAMPLE_TEXTS[:20])
    assert bpt > 1.0, f"expected >1 byte per token, got {bpt}"
    assert bpt < 20.0, f"absurdly high bytes/token: {bpt}"


def test_evaluate_returns_expected_keys(tmp_path, text_iter):
    save = tmp_path / "bpe_500"
    _trainer(save, 500, "bpe").train(text_iter)
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


# ---- SuperBPE subword pass ----


def test_superbpe_subword_pass_saves_intermediate(tmp_path, text_iter):
    save = tmp_path / "sbpe"
    # transition_size=399 keeps the superword pass role minimal — we only verify the subword pass here.
    _trainer(save, 400, "superbpe", transition_size=399).train(text_iter)
    assert (save / "subword_tokenizer.json").exists()


def test_superbpe_subword_pass_vocab_size(tmp_path, text_iter):
    save = tmp_path / "sbpe"
    _trainer(save, 400, "superbpe", transition_size=399).train(text_iter)
    subword = Tokenizer.from_file(str(save / "subword_tokenizer.json"))
    # Subword pass trains to transition_size (~400, may be slightly below due
    # to corpus size; trainer stops when no more merges are available).
    assert subword.get_vocab_size() <= 400
    assert subword.get_vocab_size() >= 256  # at least byte alphabet


# ---- SuperBPE superword pass ----


def test_superbpe_full_train_succeeds(tmp_path, text_iter):
    save = tmp_path / "sbpe_400_t300"
    _trainer(
        save,
        400,
        "superbpe",
        transition_size=300,
        max_superword_words=99,  # effectively disable cap
    ).train(text_iter)
    assert (save / "tokenizer.json").exists()


def test_superbpe_final_vocab_size(tmp_path, text_iter):
    save = tmp_path / "sbpe_400_t300"
    _trainer(save, 400, "superbpe", transition_size=300, max_superword_words=99).train(
        text_iter
    )
    tok = load_tokenizer(str(save))
    # Small corpus has enough unique pairs to reach exactly T=400.
    # If SAMPLE_TEXTS is reduced in diversity, this may need to become <= 400.
    assert tok.get_vocab_size() == 400


def test_superbpe_produces_superword(tmp_path, text_iter):
    save = tmp_path / "sbpe_600_t300"
    _trainer(save, 600, "superbpe", transition_size=300, max_superword_words=99).train(
        text_iter
    )
    tok = load_tokenizer(str(save))
    vocab = tok.get_vocab()
    # Ġ not at position 0 = internal whitespace = superword token
    superwords = [t for t in vocab if "Ġ" in t[1:]]
    assert len(superwords) > 0, "expected at least one superword token after stage 2"


def test_superbpe_decode_roundtrip(tmp_path, text_iter):
    save = tmp_path / "sbpe_400_t300"
    _trainer(save, 400, "superbpe", transition_size=300, max_superword_words=99).train(
        text_iter
    )
    tok = load_tokenizer(str(save))
    for s in ["the quick brown fox", "she sells seashells", "by the way it works"]:
        ids = tok.encode(s).ids
        assert tok.decode(ids) == s, f"roundtrip failed for: {s!r}"


def test_superbpe_load_roundtrip(tmp_path, text_iter):
    save = tmp_path / "sbpe_400_t300"
    tok_train = _trainer(
        save, 400, "superbpe", transition_size=300, max_superword_words=99
    ).train(text_iter)
    tok_load = load_tokenizer(str(save))
    for s in ["the quick brown fox", "she sells seashells", "by the way"]:
        assert tok_load.encode(s).ids == tok_train.encode(s).ids, (
            f"load roundtrip mismatch for: {s!r}"
        )


# ---- SuperBPE guards ----


def _ġ_count(tok: str) -> int:
    """Stage-1 'word count' of a token (Ġ characters + 1 if no leading Ġ)."""
    return tok.count("Ġ") + (0 if tok.startswith("Ġ") else 1)


def test_superbpe_max_superword_words_cap(tmp_path, text_iter):
    save = tmp_path / "sbpe_capped"
    _trainer(save, 600, "superbpe", transition_size=300, max_superword_words=2).train(
        text_iter
    )
    tok = load_tokenizer(str(save))
    for t in tok.get_vocab():
        assert _ġ_count(t) <= 2, f"token {t!r} exceeds 2-word cap"


def test_superbpe_no_colon_space_tokens(tmp_path, text_iter):
    # Inject some "X: Y" patterns so the algorithm is tempted to learn ":Ġ" merges.
    texts = ["foo: bar baz quux", "name: alice", "topic: animals plants"] * 50
    save = tmp_path / "sbpe_colon"
    _trainer(save, 500, "superbpe", transition_size=350, max_superword_words=99).train(
        lambda: iter(texts)
    )
    tok = load_tokenizer(str(save))
    for t in tok.get_vocab():
        assert ":Ġ" not in t, f"forbidden ':Ġ' substring in token {t!r}"


# ---- W&B curve logging ----


def test_superbpe_logs_curve_points(tmp_path, text_iter, monkeypatch):
    """Curve logging emits expected points without contacting W&B servers."""
    monkeypatch.setenv("WANDB_MODE", "disabled")

    logged: list[dict] = []
    import wandb

    real_init = wandb.init

    # real_log is set after init so we get the run's actual (no-op) log.
    real_log_holder: list = []

    def capture(d, *args, **kwargs):
        logged.append(dict(d))
        if real_log_holder:
            return real_log_holder[0](d, *args, **kwargs)

    def capture_init(*args, **kwargs):
        run = real_init(*args, **kwargs)
        # wandb.init() replaces wandb.log — capture it, then restore our wrapper.
        real_log_holder.append(wandb.log)
        wandb.log = capture
        return run

    monkeypatch.setattr(wandb, "log", capture)
    monkeypatch.setattr(wandb, "init", capture_init)

    save = tmp_path / "sbpe_wandb"
    _trainer(
        save,
        400,
        "superbpe",
        transition_size=300,
        max_superword_words=99,
        eval_num_docs=10,
        eval_every=20,
        wandb_enabled=True,
        logging_kwargs={"wandb_project": "test"},
    ).train(text_iter)
    # Expect at least one point in stage 1 region and one in stage 2 region.
    vocabs = [d["vocab_size"] for d in logged if "vocab_size" in d]
    assert any(v < 300 for v in vocabs), f"no stage-1 curve points: {vocabs}"
    assert any(v > 300 for v in vocabs), f"no stage-2 curve points: {vocabs}"
    # Forced point at transition_size present (within a few units due to
    # corpus-driven actual stage-1 vocab size).
    assert any(abs(v - 300) <= 5 for v in vocabs), (
        f"no transition point near 300: {vocabs}"
    )
    # Final point present.
    assert any(v == 400 for v in vocabs), f"no final point at 400: {vocabs}"
    # Every logged point has bytes_per_token.
    for d in logged:
        if "vocab_size" in d:
            assert "bytes_per_token" in d, f"missing bytes_per_token in {d}"


# ---- encode_corpus_to_chunks ----


def test_encode_corpus_to_chunks_aggregates_by_token_tuple(tmp_path, text_iter):
    """Helper produces a dict mapping post-encode token-tuples to frequencies."""
    # Train a small tokenizer to encode with.
    save = tmp_path / "small_bpe"
    tok_train = _trainer(save, 300, "bpe").train(text_iter)

    # Drive the helper.
    chunks = encode_corpus_to_chunks(tok_train, text_iter, batch_size=10)

    # Shape: dict[tuple[str,...], int], non-empty, all counts positive.
    assert isinstance(chunks, dict)
    assert len(chunks) > 0
    for k, v in chunks.items():
        assert isinstance(k, tuple)
        assert all(isinstance(s, str) for s in k)
        assert v > 0

    # Aggregation correctness: total token-occurrences = sum of all docs'
    # token counts after encoding.
    expected_total = sum(
        len(tok_train.encode(doc, add_special_tokens=False).tokens)
        for doc in text_iter()
    )
    counted_total = sum(len(k) * v for k, v in chunks.items())
    assert counted_total == expected_total


def test_encode_corpus_to_chunks_dedups_identical_docs(tmp_path, text_iter):
    """Identical input docs collapse to one chunk with count = num occurrences."""
    # Train a small tokenizer (any working one — we just need it to encode).
    save = tmp_path / "small_bpe"
    tok = _trainer(save, 300, "bpe").train(text_iter)

    docs = ["hello world", "hello world", "hello world", "different"]
    chunks = encode_corpus_to_chunks(tok, lambda: iter(docs))

    # "hello world" → one chunk with count 3; "different" → one chunk with count 1.
    assert len(chunks) == 2
    assert max(chunks.values()) == 3
    assert min(chunks.values()) == 1
