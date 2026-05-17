"""Tests for src/data/bpe.py — pure-Python BPE trainer."""

from src.data.bpe import BpeTrainer, _BYTE_TO_UNICODE, _UNICODE_TO_BYTE, _byte_encode


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
