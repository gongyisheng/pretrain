"""Tests for src/data/bpe.py — pure-Python BPE trainer."""

import pytest

from src.data.bpe import (
    BpeTrainer,
    _BYTE_TO_UNICODE,
    _UNICODE_TO_BYTE,
    _byte_encode,
    _pretokenize,
)


def test_bpe_module_exports_trainer():
    assert BpeTrainer is not None
    assert callable(BpeTrainer)


def test_byte_to_unicode_is_256_entries():
    assert len(_BYTE_TO_UNICODE) == 256
    assert set(_BYTE_TO_UNICODE.keys()) == set(range(256))


def test_byte_to_unicode_is_invertible():
    for b, c in _BYTE_TO_UNICODE.items():
        assert _UNICODE_TO_BYTE[c] == b


def test_byte_encode_space_is_gdot():
    # GPT-2 byte-level: 0x20 (space) maps to "Ġ".
    assert _byte_encode(" ") == "Ġ"


def test_byte_encode_matches_hf_ascii():
    from tokenizers import pre_tokenizers

    bl = pre_tokenizers.ByteLevel(add_prefix_space=False)
    for s in ["hello", "the quick brown fox", "a b c"]:
        hf_out = "".join(piece for piece, _ in bl.pre_tokenize_str(s))
        assert _byte_encode(s) == hf_out, f"mismatch for {s!r}"


def test_byte_encode_matches_hf_unicode():
    from tokenizers import pre_tokenizers

    bl = pre_tokenizers.ByteLevel(add_prefix_space=False)
    for s in ["über", "naïve", "café", "日本語"]:
        hf_out = "".join(piece for piece, _ in bl.pre_tokenize_str(s))
        assert _byte_encode(s) == hf_out, f"mismatch for {s!r}"


def test_pretokenize_bpe_mode_splits_on_whitespace():
    out = _pretokenize("the quick brown", mode="bpe")
    # Each piece is byte-encoded; spaces become Ġ-prefix on the following word.
    assert out == ["the", "Ġquick", "Ġbrown"]


def test_pretokenize_bpe_mode_digit_grouping():
    # Digits split into groups of 3 from the right.
    out = _pretokenize("123456789", mode="bpe")
    assert out == ["123", "456", "789"]
    out = _pretokenize("1234", mode="bpe")
    assert out == ["1", "234"]


def test_pretokenize_bpe_mode_punctuation_separate():
    out = _pretokenize("hello, world!", mode="bpe")
    # ", " stays grouped per the paper regex (`[^\s\p{L}\p{N}]+` with leading space),
    # "!" trails.
    assert "hello" in out
    assert "Ġworld" in out


def test_pretokenize_bytelevel_mode_is_single_chunk():
    s = "the quick brown"
    out = _pretokenize(s, mode="bytelevel")
    assert len(out) == 1
    assert out[0] == "theĠquickĠbrown"


def test_pretokenize_matches_hf_byte_for_byte():
    # The "bpe" mode must produce the same pieces (in order, after byte-encoding)
    # as the HF pretokenizer used by SuperBPE stage 1.
    from tokenizers import Regex, pre_tokenizers

    hf_pretok = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(r" ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"),
                behavior="isolated",
            ),
            pre_tokenizers.Split(Regex(r"(?=(\d{3})+(?!\d))"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    for s in [
        "the quick brown fox",
        "she sells 12345 seashells",
        "naïve café 999",
        "no spaces here",
    ]:
        hf_pieces = [piece for piece, _ in hf_pretok.pre_tokenize_str(s)]
        assert _pretokenize(s, mode="bpe") == hf_pieces, f"mismatch for {s!r}"


def test_pretokenize_unknown_mode_raises():
    with pytest.raises(ValueError, match="pretokenizer"):
        _pretokenize("hello", mode="not_a_mode")
