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


def test_init_pair_state_parallel_matches_serial():
    """Parallel pair-counting must produce byte-identical state to serial."""
    # Build a moderate corpus of byte-encoded chunks so each worker sees real work.
    chunks: dict[tuple[str, ...], int] = {}
    for i in range(200):
        text = f"sample document number {i} with some repeated content"
        toks = tuple(_byte_encode(text))
        chunks[toks] = chunks.get(toks, 0) + 1

    s_s, w_s, pc_s, wtu_s = _init_pair_state(chunks, merges_to_replay=[], n_workers=1)
    s_p, w_p, pc_p, wtu_p = _init_pair_state(
        chunks, merges_to_replay=[], n_workers=4, chunks_per_task=20
    )

    assert s_s == s_p
    assert w_s == w_p
    assert pc_s == pc_p
    assert wtu_s == wtu_p


def test_select_best_pair_picks_max_count():
    vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
    pair_counts = Counter({("a", "b"): 5, ("c", "d"): 3})
    heap = [_make_heap_entry(p, c, vocab) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    pair, cnt = _select_best_pair(heap, pair_counts)
    assert pair == ("a", "b")
    assert cnt == 5


def test_select_best_pair_tie_break_smaller_id_wins():
    # Equal count → pick the pair with smaller token IDs (matches HF's tie-break:
    # ascending on (id_a, id_b)). With lex-sorted alphabet IDs, 'a'<'b'<'c'<'d',
    # so ('a','b') has the smallest ID pair.
    vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
    pair_counts = Counter({("a", "b"): 3, ("a", "c"): 3, ("a", "d"): 3})
    heap = [_make_heap_entry(p, c, vocab) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    pair, _ = _select_best_pair(heap, pair_counts)
    assert pair == ("a", "b")


def test_select_best_pair_skips_stale():
    # Heap has stale entry for ("a","b") at count=10; current count is 2.
    vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
    pair_counts = Counter({("a", "b"): 2, ("c", "d"): 5})
    heap = [
        _make_heap_entry(("a", "b"), 10, vocab),  # stale (count too high)
        _make_heap_entry(("a", "b"), 2, vocab),  # current
        _make_heap_entry(("c", "d"), 5, vocab),
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
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3}
    chunks = {("a", "b", "c"): 1, ("a", "b"): 1}
    syms, weights, pair_counts, where = _init_pair_state(chunks, [])
    heap = [_make_heap_entry(p, c, vocab) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    _apply_merge(syms, weights, pair_counts, where, heap, ("a", "b"), "ab", vocab)
    # All ("a","b") occurrences collapsed.
    # _init_pair_state sorts chunks: ("a","b") < ("a","b","c"), so chunk 0 = ["a","b"].
    assert syms == [["ab"], ["ab", "c"]]


def test_apply_merge_updates_pair_counts():
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3}
    chunks = {("a", "b", "c"): 1}
    syms, weights, pair_counts, where = _init_pair_state(chunks, [])
    heap = [_make_heap_entry(p, c, vocab) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    _apply_merge(syms, weights, pair_counts, where, heap, ("a", "b"), "ab", vocab)
    # (a,b) gone, (b,c) gone, (ab,c) new.
    assert ("a", "b") not in pair_counts
    assert ("b", "c") not in pair_counts
    assert pair_counts[("ab", "c")] == 1


def test_apply_merge_respects_chunk_weight():
    vocab = {"a": 0, "b": 1, "ab": 2}
    chunks = {("a", "b"): 5}
    syms, weights, pair_counts, where = _init_pair_state(chunks, [])
    heap = [_make_heap_entry(p, c, vocab) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    _apply_merge(syms, weights, pair_counts, where, heap, ("a", "b"), "ab", vocab)
    # (a,b) had count 5; after merge it disappears, no new pairs from a 1-symbol chunk.
    assert ("a", "b") not in pair_counts
    assert syms == [["ab"]]


def test_apply_merge_overlapping_pairs_left_to_right():
    # "a a a" → merge (a,a): consume left pair first, leaving "aa a", not "a aa".
    vocab = {"a": 0, "aa": 1}
    chunks = {("a", "a", "a"): 1}
    syms, weights, pair_counts, where = _init_pair_state(chunks, [])
    heap = [_make_heap_entry(p, c, vocab) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    _apply_merge(syms, weights, pair_counts, where, heap, ("a", "a"), "aa", vocab)
    assert syms == [["aa", "a"]]
    assert pair_counts[("aa", "a")] == 1


SAMPLE_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
    "to be or not to be that is the question",
] * 20


def _corpus():
    return iter(SAMPLE_TEXTS)


def test_bpe_train_returns_vocab_and_merges():
    vocab, merges = BpeTrainer(vocab_size=300, n_workers=1).train(_corpus)
    assert isinstance(vocab, dict)
    assert isinstance(merges, list)
    # All merge symbols must be in vocab.
    for a, b in merges:
        assert a in vocab and b in vocab
        assert a + b in vocab


def test_bpe_vocab_size_respected():
    vocab, _ = BpeTrainer(vocab_size=300, n_workers=1).train(_corpus)
    assert len(vocab) <= 300


def test_bpe_vocab_ids_contiguous_from_zero():
    vocab, _ = BpeTrainer(vocab_size=300, n_workers=1).train(_corpus)
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_bpe_vocab_size_too_small_raises():
    # 256 byte alphabet + 1 special = 257 minimum.
    with pytest.raises(ValueError, match="vocab_size"):
        BpeTrainer(vocab_size=100, n_workers=1)


def test_bpe_unknown_pretokenizer_raises():
    with pytest.raises(ValueError, match="pretokenizer"):
        BpeTrainer(vocab_size=300, pretokenizer="not_a_mode", n_workers=1)


def test_bpe_empty_corpus_raises():
    with pytest.raises(ValueError, match="no trainable chunks"):
        BpeTrainer(vocab_size=300, n_workers=1).train(lambda: iter([]))


def test_bpe_output_loads_into_hf():
    """Generated (vocab, merges) must build a working HF Tokenizer."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocab, merges = BpeTrainer(vocab_size=300, n_workers=1).train(_corpus)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    s = "the quick brown fox"
    ids = tok.encode(s).ids
    assert len(ids) > 0
    assert tok.decode(ids) == s


def _train_hf(corpus_iter, vocab_size):
    """Train HF BpeTrainer for the same vocab_size on the same corpus.

    Uses ByteLevel(use_regex=False) to disable HF's built-in whitespace split,
    matching our ``pretokenizer="bytelevel"`` mode (whole-document chunks).
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus_iter(), trainer=trainer)
    import json

    data = json.loads(tok.to_str())
    vocab = data["model"]["vocab"]
    merges = [
        tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
        for m in data["model"]["merges"]
    ]
    return vocab, merges


def test_bpe_parity_with_hf_ascii():
    py_vocab, py_merges = BpeTrainer(
        vocab_size=400, pretokenizer="bytelevel", n_workers=1
    ).train(_corpus)
    hf_vocab, hf_merges = _train_hf(_corpus, vocab_size=400)
    assert py_merges == hf_merges, (
        f"first divergence at index "
        f"{next((i for i, (a, b) in enumerate(zip(py_merges, hf_merges)) if a != b), '?')}"
    )
    # Vocab keys should match exactly; ID assignment may differ, but the
    # set of tokens must be identical.
    assert set(py_vocab) == set(hf_vocab)


UNICODE_TEXTS = [
    "über naïve cafés au lait sont délicieux",
    "naïve über alles résumé façade",
    "日本語のテスト 漢字 ひらがな",
] * 30


def test_bpe_parity_with_hf_unicode():
    py_vocab, py_merges = BpeTrainer(
        vocab_size=400, pretokenizer="bytelevel", n_workers=1
    ).train(lambda: iter(UNICODE_TEXTS))
    hf_vocab, hf_merges = _train_hf(lambda: iter(UNICODE_TEXTS), vocab_size=400)
    assert py_merges == hf_merges
    assert set(py_vocab) == set(hf_vocab)


def test_merge_filter_vetoes_specific_pair():
    # Reject any merge whose result contains "e".
    def no_e(a, b, merged):
        return "e" not in merged

    trainer = BpeTrainer(vocab_size=300, n_workers=1, merge_filter=no_e)
    vocab, _ = trainer.train(_corpus)
    special = set(trainer.special_tokens)
    for tok in vocab:
        if len(tok) > 1 and tok not in special:  # multi-char tokens are merge results
            assert "e" not in tok, f"vetoed letter leaked into {tok!r}"


def test_merge_filter_early_exit_when_all_vetoed():
    """Filter rejecting everything → trainer exits with seed-only vocab."""

    def reject_all(a, b, merged):
        return False

    vocab, merges = BpeTrainer(
        vocab_size=300, n_workers=1, merge_filter=reject_all
    ).train(_corpus)
    # 256 byte alphabet + 1 special token. No merges accepted.
    assert len(vocab) == 257
    assert merges == []


def test_merge_filter_called_with_string_args():
    """Filter receives (a, b, a+b) as byte-level unicode strings."""
    calls = []

    def record(a, b, merged):
        calls.append((a, b, merged))
        return True

    BpeTrainer(vocab_size=270, n_workers=1, merge_filter=record).train(_corpus)
    assert len(calls) > 0
    for a, b, merged in calls:
        assert isinstance(a, str) and isinstance(b, str) and isinstance(merged, str)
        assert a + b == merged


def test_progress_callback_fires_every_n():
    calls = []

    def cb(vocab_size, vocab, merges):
        calls.append(vocab_size)

    BpeTrainer(
        vocab_size=400,
        n_workers=1,
        progress_callback=cb,
        progress_every=50,
    ).train(_corpus)
    # Vocab grows from ~257 (specials+alphabet) to 400.
    # progress_every=50 means callback fires when vocab_size hits a multiple of 50.
    assert all(v % 50 == 0 for v in calls), f"non-multiple firing: {calls}"
    assert len(calls) >= 1


def test_progress_callback_receives_current_state():
    captured = []

    def cb(vs, vocab, merges):
        captured.append((vs, len(vocab), len(merges)))

    BpeTrainer(
        vocab_size=400,
        n_workers=1,
        progress_callback=cb,
        progress_every=50,
    ).train(_corpus)
    # vocab_size arg always equals len(vocab) at callback time.
    for vs, n_vocab, _ in captured:
        assert vs == n_vocab


def test_resume_extends_to_larger_vocab():
    # Train to 300.
    v1, m1 = BpeTrainer(vocab_size=300, n_workers=1).train(_corpus)
    # Resume to 400 with the same corpus.
    v2, m2 = BpeTrainer(
        vocab_size=400,
        n_workers=1,
        initial_vocab=v1,
        initial_merges=m1,
    ).train(_corpus)
    # All of v1 must persist with the same IDs.
    for tok, tid in v1.items():
        assert v2[tok] == tid
    # First |m1| merges must be unchanged.
    assert m2[: len(m1)] == m1


def test_resume_equals_fresh_train():
    """Fresh train to 400 == train-to-300 then resume to 400."""
    fresh_v, fresh_m = BpeTrainer(vocab_size=400, n_workers=1).train(_corpus)
    v1, m1 = BpeTrainer(vocab_size=300, n_workers=1).train(_corpus)
    resumed_v, resumed_m = BpeTrainer(
        vocab_size=400,
        n_workers=1,
        initial_vocab=v1,
        initial_merges=m1,
    ).train(_corpus)
    assert resumed_m == fresh_m
    assert set(resumed_v) == set(fresh_v)


def test_resume_inconsistent_initial_raises():
    # initial_merges references a token not in initial_vocab.
    with pytest.raises(ValueError, match="initial_merges references unknown"):
        BpeTrainer(
            vocab_size=400,
            n_workers=1,
            initial_vocab={"<|endoftext|>": 0, "a": 1, "b": 2},
            initial_merges=[("a", "z")],  # "z" not in vocab
        )


def test_resume_one_without_other_raises():
    with pytest.raises(ValueError, match="both be provided or both None"):
        BpeTrainer(vocab_size=400, n_workers=1, initial_vocab={"a": 0})
    with pytest.raises(ValueError, match="both be provided or both None"):
        BpeTrainer(vocab_size=400, n_workers=1, initial_merges=[("a", "b")])


def test_resume_initial_too_large_raises():
    big_vocab = {f"t{i}": i for i in range(500)}
    with pytest.raises(ValueError, match="initial_vocab size .* > vocab_size"):
        BpeTrainer(
            vocab_size=400,
            n_workers=1,
            initial_vocab=big_vocab,
            initial_merges=[],
        )


def test_resume_initial_vocab_non_contiguous_raises():
    # IDs must be 0..N-1.
    with pytest.raises(ValueError, match="contiguous from 0"):
        BpeTrainer(
            vocab_size=400,
            n_workers=1,
            initial_vocab={"a": 0, "b": 2, "c": 3},  # gap at 1
            initial_merges=[],
        )


def test_build_chunks_parallel_matches_serial():
    serial = _build_chunks(_corpus, mode="bpe", n_workers=1)
    parallel = _build_chunks(_corpus, mode="bpe", n_workers=4, batch_size=10)
    assert serial == parallel


def test_bpe_parity_serial_vs_parallel():
    """Same corpus + vocab_size → identical (vocab, merges) regardless of worker count."""
    v_serial, m_serial = BpeTrainer(vocab_size=400, n_workers=1).train(_corpus)
    v_parallel, m_parallel = BpeTrainer(
        vocab_size=400,
        n_workers=4,
        batch_size=10,
    ).train(_corpus)
    assert m_serial == m_parallel
    assert v_serial == v_parallel


def test_bpe_n_workers_default_is_safe():
    """n_workers=None defaults to a sane positive integer; train succeeds."""
    BpeTrainer(vocab_size=300, n_workers=None).train(_corpus)


def test_bpe_state_seed_assigns_chunk_ids_in_tuple_order():
    """Chunks must be sorted by tuple-of-IDs, matching today's
    sorted(chunks.items(), key=lambda kv: kv[0]).
    """
    from src.data.bpe_native import BpeState

    # IDs assigned so that lex-of-id order == lex-of-str order on the symbols used.
    symbol_table = {"a": 0, "b": 1, "c": 2}
    chunks = {("b", "c"): 1, ("a", "b"): 2, ("a", "c"): 3}

    state = BpeState()
    state.seed(chunks, symbol_table)

    # Sorted tuples: ("a","b"), ("a","c"), ("b","c") → chunk_ids 0,1,2.
    assert state.get_chunk_symbols(0) == [0, 1]
    assert state.get_chunk_symbols(1) == [0, 2]
    assert state.get_chunk_symbols(2) == [1, 2]
    assert state.get_chunk_weight(0) == 2
    assert state.get_chunk_weight(1) == 3
    assert state.get_chunk_weight(2) == 1
    assert state.num_chunks() == 3


def test_bpe_state_seed_unknown_symbol_raises():
    from src.data.bpe_native import BpeState

    symbol_table = {"a": 0, "b": 1}
    chunks = {("a", "z"): 1}  # 'z' not in symbol_table

    state = BpeState()
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        state.seed(chunks, symbol_table)


def test_bpe_state_build_initial_pairs_counts_adjacent():
    """Counts adjacent-pair occurrences weighted by chunk frequency."""
    from src.data.bpe_native import BpeState

    symbol_table = {"a": 0, "b": 1, "c": 2}
    chunks = {("a", "b", "c"): 2, ("a", "b"): 3}

    state = BpeState()
    state.seed(chunks, symbol_table)
    pairs = state.build_initial_pairs()

    # Convert to dict for assertion. Each entry is (a_id, b_id, count).
    by_pair = {(a, b): c for a, b, c in pairs}
    # (a,b) appears in both chunks: count = 2 + 3 = 5.
    # (b,c) appears in first: count = 2.
    assert by_pair[(0, 1)] == 5
    assert by_pair[(1, 2)] == 2
    assert len(by_pair) == 2


def test_bpe_state_pair_chunks_indexes_membership():
    """pair_chunks returns the chunk_ids where (a,b) appears."""
    from src.data.bpe_native import BpeState

    symbol_table = {"a": 0, "b": 1, "c": 2}
    chunks = {("a", "b"): 1, ("a", "c"): 1}

    state = BpeState()
    state.seed(chunks, symbol_table)
    state.build_initial_pairs()

    ab = sorted(state.pair_chunks(0, 1))
    ac = sorted(state.pair_chunks(0, 2))
    assert len(ab) == 1
    assert len(ac) == 1
    assert ab != ac


def test_bpe_state_pair_count_matches_initial_pairs():
    """pair_count(a,b) reflects what build_initial_pairs returned."""
    from src.data.bpe_native import BpeState

    symbol_table = {"a": 0, "b": 1}
    state = BpeState()
    state.seed({("a", "b"): 7}, symbol_table)
    state.build_initial_pairs()

    assert state.pair_count(0, 1) == 7
    assert state.pair_count(1, 0) == 0  # never observed


def test_bpe_state_build_initial_pairs_dedupes_pair_within_chunk():
    """If (a,b) appears multiple times in one chunk, the chunk_id is listed
    in where_[(a,b)] exactly once, but pair_count counts every occurrence
    weighted by chunk weight."""
    from src.data.bpe_native import BpeState

    symbol_table = {"a": 0, "b": 1}
    chunks = {("a", "b", "a", "b"): 3}

    state = BpeState()
    state.seed(chunks, symbol_table)
    state.build_initial_pairs()

    # Two occurrences of (a,b) in a single chunk with weight 3.
    assert state.pair_count(0, 1) == 6
    # But the chunk_id appears only once in pair_chunks.
    assert state.pair_chunks(0, 1) == [0]


def test_bpe_state_replay_single_merge():
    """Replaying (a,b)→ab on chunk ("a","b","c") gives [ab_id, c_id]."""
    from src.data.bpe_native import BpeState

    symbol_table = {"a": 0, "b": 1, "c": 2, "ab": 3}
    state = BpeState()
    state.seed({("a", "b", "c"): 1}, symbol_table)
    state.replay_merges([(0, 1, 3)])  # (a_id, b_id, merged_id)

    assert state.get_chunk_symbols(0) == [3, 2]


def test_bpe_state_replay_multi_merge_in_order():
    """Replaying (a,b)→ab then (ab,c)→abc collapses to a single token."""
    from src.data.bpe_native import BpeState

    symbol_table = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}
    state = BpeState()
    state.seed({("a", "b", "c"): 1}, symbol_table)
    state.replay_merges([(0, 1, 3), (3, 2, 4)])

    assert state.get_chunk_symbols(0) == [4]


def test_bpe_state_replay_left_to_right_overlap():
    """Replaying (a,a)→aa on ("a","a","a") gives [aa, a], not [a, aa]."""
    from src.data.bpe_native import BpeState

    symbol_table = {"a": 0, "aa": 1}
    state = BpeState()
    state.seed({("a", "a", "a"): 1}, symbol_table)
    state.replay_merges([(0, 0, 1)])

    assert state.get_chunk_symbols(0) == [1, 0]


def test_bpe_state_replay_parallel_matches_serial(monkeypatch):
    """OMP_NUM_THREADS=1 vs default must produce identical final state."""
    from src.data.bpe_native import BpeState
    from src.data.bpe import _byte_encode

    symbol_table = {
        c: i
        for i, c in enumerate(
            sorted(
                set(_byte_encode("sample document number 0 with some repeated content"))
            )
        )
    }
    # Build a moderate corpus that exercises multi-chunk parallelism.
    chunks: dict[tuple[str, ...], int] = {}
    for i in range(200):
        toks = tuple(
            _byte_encode(f"sample document number {i} with some repeated content")
        )
        for ch in toks:
            symbol_table.setdefault(ch, len(symbol_table))
        chunks[toks] = chunks.get(toks, 0) + 1

    # Pick a merge that will fire frequently: (' ', 's') if both present.
    space_id = symbol_table.get("Ġ")
    s_id = symbol_table.get("s")
    if space_id is None or s_id is None:
        pytest.skip("expected byte-encoded space and 's' in symbol_table")
    merged_str = "Ġs"
    merged_id = len(symbol_table)
    symbol_table[merged_str] = merged_id

    s1 = BpeState()
    s1.seed(chunks, symbol_table)
    s1.set_num_threads(1)
    s1.replay_merges([(space_id, s_id, merged_id)])

    s2 = BpeState()
    s2.seed(chunks, symbol_table)
    s2.set_num_threads(4)
    s2.replay_merges([(space_id, s_id, merged_id)])

    n = s1.num_chunks()
    assert n == s2.num_chunks()
    for i in range(n):
        assert s1.get_chunk_symbols(i) == s2.get_chunk_symbols(i)
