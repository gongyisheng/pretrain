"""Tests for src/data/bpe.py — pure-Python BPE trainer."""

import heapq
from collections import Counter

import pytest

from src.data.bpe import (
    BpeTrainer,
    _BYTE_TO_UNICODE,
    _UNICODE_TO_BYTE,
    _apply_merge,
    _build_chunks,
    _byte_encode,
    _init_pair_state,
    _make_heap_entry,
    _pretokenize,
    _select_best_pair,
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


def test_build_chunks_dedups_by_symbol_tuple():
    docs = ["the the", "the"]
    chunks = _build_chunks(lambda: iter(docs), mode="bpe", n_workers=1)
    # "the" appears 3 times: once at start, once after Ġ, then standalone again.
    # _pretokenize("the the") = ["the", "Ġthe"]. _pretokenize("the") = ["the"].
    # So chunks = {("t","h","e"): 2, ("Ġ","t","h","e"): 1}.
    assert chunks[tuple("the")] == 2
    assert chunks[tuple("Ġthe")] == 1


def test_build_chunks_bytelevel_one_chunk_per_doc():
    docs = ["abc", "abc", "xyz"]
    chunks = _build_chunks(lambda: iter(docs), mode="bytelevel", n_workers=1)
    assert chunks[("a", "b", "c")] == 2
    assert chunks[("x", "y", "z")] == 1


def test_build_chunks_empty_corpus_returns_empty_dict():
    chunks = _build_chunks(lambda: iter([]), mode="bpe", n_workers=1)
    assert chunks == {}


def test_init_pair_state_counts_adjacent_pairs():
    chunks = {("a", "b", "c"): 2, ("a", "b"): 3}
    symbols, weights, pair_counts, where = _init_pair_state(chunks, merges_to_replay=[])
    # (a,b) appears in both chunks: count = 2*1 + 3*1 = 5.
    # (b,c) appears in first chunk: count = 2*1 = 2.
    assert pair_counts[("a", "b")] == 5
    assert pair_counts[("b", "c")] == 2


def test_init_pair_state_where_to_update_indexes_chunks():
    chunks = {("a", "b"): 1, ("a", "c"): 1}
    symbols, weights, pair_counts, where = _init_pair_state(chunks, merges_to_replay=[])
    ab_chunks = where[("a", "b")]
    ac_chunks = where[("a", "c")]
    assert len(ab_chunks) == 1
    assert len(ac_chunks) == 1
    assert ab_chunks != ac_chunks


def test_init_pair_state_symbols_per_chunk_are_lists():
    chunks = {("a", "b", "c"): 1}
    symbols, weights, _, _ = _init_pair_state(chunks, merges_to_replay=[])
    assert symbols == [["a", "b", "c"]]


def test_init_pair_state_chunk_counts_track_input_weights():
    chunks = {("a",): 7}
    _, weights, _, _ = _init_pair_state(chunks, merges_to_replay=[])
    assert weights == [7]


def test_init_pair_state_chunk_id_assignment_is_deterministic():
    chunks = {("b",): 1, ("a",): 1, ("c",): 1}
    s1, _, _, _ = _init_pair_state(chunks, merges_to_replay=[])
    s2, _, _, _ = _init_pair_state(chunks, merges_to_replay=[])
    assert s1 == s2  # same input → same chunk ordering


def test_init_pair_state_replays_merges():
    chunks = {("a", "b", "c"): 1}
    symbols, _, pair_counts, _ = _init_pair_state(chunks, merges_to_replay=[("a", "b")])
    assert symbols == [["ab", "c"]]
    assert pair_counts == Counter({("ab", "c"): 1})


def test_init_pair_state_replay_multiple_merges_in_order():
    chunks = {("a", "b", "c"): 1}
    symbols, _, _, _ = _init_pair_state(
        chunks, merges_to_replay=[("a", "b"), ("ab", "c")]
    )
    assert symbols == [["abc"]]


def test_select_best_pair_picks_max_count():
    pair_counts = Counter({("a", "b"): 5, ("c", "d"): 3})
    heap = [_make_heap_entry(p, c) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    pair, cnt = _select_best_pair(heap, pair_counts)
    assert pair == ("a", "b")
    assert cnt == 5


def test_select_best_pair_tie_break_lex_larger_wins():
    # Equal count → pick the lex-LARGER pair (matches HF BinaryHeap max-heap on pair tuple).
    pair_counts = Counter({("a", "b"): 3, ("a", "c"): 3, ("a", "d"): 3})
    heap = [_make_heap_entry(p, c) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    pair, _ = _select_best_pair(heap, pair_counts)
    assert pair == ("a", "d")


def test_select_best_pair_skips_stale():
    # Heap has stale entry for ("a","b") at count=10; current count is 2.
    pair_counts = Counter({("a", "b"): 2, ("c", "d"): 5})
    heap = [
        _make_heap_entry(("a", "b"), 10),  # stale (count too high)
        _make_heap_entry(("a", "b"), 2),  # current
        _make_heap_entry(("c", "d"), 5),
    ]
    heapq.heapify(heap)
    pair, cnt = _select_best_pair(heap, pair_counts)
    # Stale entry popped first → ignored; then (c,d):5 is chosen over (a,b):2.
    assert pair == ("c", "d")
    assert cnt == 5


def test_select_best_pair_returns_none_when_heap_empty():
    heap: list = []
    pair_counts: Counter = Counter()
    assert _select_best_pair(heap, pair_counts) == (None, 0)


def test_apply_merge_rewrites_symbols():
    chunks = {("a", "b", "c"): 1, ("a", "b"): 1}
    syms, weights, pair_counts, where = _init_pair_state(chunks, [])
    heap = [_make_heap_entry(p, c) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    _apply_merge(syms, weights, pair_counts, where, heap, ("a", "b"), "ab")
    # All ("a","b") occurrences collapsed.
    # _init_pair_state sorts chunks: ("a","b") < ("a","b","c"), so chunk 0 = ["a","b"].
    assert syms == [["ab"], ["ab", "c"]]


def test_apply_merge_updates_pair_counts():
    chunks = {("a", "b", "c"): 1}
    syms, weights, pair_counts, where = _init_pair_state(chunks, [])
    heap = [_make_heap_entry(p, c) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    _apply_merge(syms, weights, pair_counts, where, heap, ("a", "b"), "ab")
    # (a,b) gone, (b,c) gone, (ab,c) new.
    assert ("a", "b") not in pair_counts
    assert ("b", "c") not in pair_counts
    assert pair_counts[("ab", "c")] == 1


def test_apply_merge_respects_chunk_weight():
    chunks = {("a", "b"): 5}
    syms, weights, pair_counts, where = _init_pair_state(chunks, [])
    heap = [_make_heap_entry(p, c) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    _apply_merge(syms, weights, pair_counts, where, heap, ("a", "b"), "ab")
    # (a,b) had count 5; after merge it disappears, no new pairs from a 1-symbol chunk.
    assert ("a", "b") not in pair_counts
    assert syms == [["ab"]]


def test_apply_merge_overlapping_pairs_left_to_right():
    # "a a a" → merge (a,a): consume left pair first, leaving "aa a", not "a aa".
    chunks = {("a", "a", "a"): 1}
    syms, weights, pair_counts, where = _init_pair_state(chunks, [])
    heap = [_make_heap_entry(p, c) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    _apply_merge(syms, weights, pair_counts, where, heap, ("a", "a"), "aa")
    assert syms == [["aa", "a"]]
    assert pair_counts[("aa", "a")] == 1
