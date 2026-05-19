"""Tests for src/data/bpe.py — pure-Python BPE trainer (now backed by
the C++ BpeEngine extension for hot loops)."""

import pytest

from src.data.bpe import (
    BpeEngine,
    BpeTrainer,
    _BYTE_TO_UNICODE,
    _UNICODE_TO_BYTE,
    _build_chunks,
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


def test_build_chunks_dedups_by_token_tuple():
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


def test_bpe_state_feed_chunk_id_deterministic():
    """Same input dict → same chunk_id assignment across two feed() calls."""

    vocab = {"a": 0, "b": 1, "c": 2}
    chunks = {("b",): 1, ("a",): 1, ("c",): 1}

    s1 = BpeEngine()
    s1.feed(chunks, vocab)
    s2 = BpeEngine()
    s2.feed(chunks, vocab)

    tokens1 = [s1.get_chunk_tokens(i) for i in range(s1.get_num_chunks())]
    tokens2 = [s2.get_chunk_tokens(i) for i in range(s2.get_num_chunks())]
    assert tokens1 == tokens2
    # Sorted by tuple: a < b < c.
    assert tokens1 == [[0], [1], [2]]


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
    # All merge tokens must be in vocab.
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


def test_bpe_state_feed_assigns_chunk_ids_in_tuple_order():
    """Chunks must be sorted by tuple-of-IDs, matching today's
    sorted(chunks.items(), key=lambda kv: kv[0]).
    """

    # IDs assigned so that lex-of-id order == lex-of-str order on the tokens used.
    vocab = {"a": 0, "b": 1, "c": 2}
    chunks = {("b", "c"): 1, ("a", "b"): 2, ("a", "c"): 3}

    state = BpeEngine()
    state.feed(chunks, vocab)

    # Sorted tuples: ("a","b"), ("a","c"), ("b","c") → chunk_ids 0,1,2.
    assert state.get_chunk_tokens(0) == [0, 1]
    assert state.get_chunk_tokens(1) == [0, 2]
    assert state.get_chunk_tokens(2) == [1, 2]
    assert state.get_chunk_count(0) == 2
    assert state.get_chunk_count(1) == 3
    assert state.get_chunk_count(2) == 1
    assert state.get_num_chunks() == 3


def test_bpe_state_feed_unknown_symbol_raises():

    vocab = {"a": 0, "b": 1}
    chunks = {("a", "z"): 1}  # 'z' not in vocab

    state = BpeEngine()
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        state.feed(chunks, vocab)


def test_bpe_state_feed_pair_index_counts_adjacent():
    """Counts adjacent-pair occurrences weighted by chunk frequency."""

    vocab = {"a": 0, "b": 1, "c": 2}
    chunks = {("a", "b", "c"): 2, ("a", "b"): 3}

    state = BpeEngine()
    state.feed(chunks, vocab)
    pairs = state.list_pairs()

    # Convert to dict for assertion. Each entry is (a_id, b_id, count).
    by_pair = {(a, b): c for a, b, c in pairs}
    # (a,b) appears in both chunks: count = 2 + 3 = 5.
    # (b,c) appears in first: count = 2.
    assert by_pair[(0, 1)] == 5
    assert by_pair[(1, 2)] == 2
    assert len(by_pair) == 2


def test_bpe_state_feed_chunk_membership():
    """get_chunks_by_pair returns the chunk_ids where (a,b) appears."""

    vocab = {"a": 0, "b": 1, "c": 2}
    chunks = {("a", "b"): 1, ("a", "c"): 1}

    state = BpeEngine()
    state.feed(chunks, vocab)

    ab = sorted(state.get_chunks_by_pair(0, 1))
    ac = sorted(state.get_chunks_by_pair(0, 2))
    assert len(ab) == 1
    assert len(ac) == 1
    assert ab != ac


def test_bpe_state_pair_count_matches_initial_pairs():
    """get_pair_count(a,b) reflects what feed built into the pair index."""

    vocab = {"a": 0, "b": 1}
    state = BpeEngine()
    state.feed({("a", "b"): 7}, vocab)

    assert state.get_pair_count(0, 1) == 7
    assert state.get_pair_count(1, 0) == 0  # never observed


def test_bpe_state_feed_pair_index_dedupes_pair_within_chunk():
    """If (a,b) appears multiple times in one chunk, the chunk_id is listed
    in pairs_[(a,b)].chunk_ids exactly once, but get_pair_count counts every
    occurrence weighted by chunk count."""

    vocab = {"a": 0, "b": 1}
    chunks = {("a", "b", "a", "b"): 3}

    state = BpeEngine()
    state.feed(chunks, vocab)

    # Two occurrences of (a,b) in a single chunk with weight 3.
    assert state.get_pair_count(0, 1) == 6
    # But the chunk_id appears only once in get_chunks_by_pair.
    assert state.get_chunks_by_pair(0, 1) == [0]


def test_bpe_state_replay_single_merge():
    """Replaying (a,b)→ab on chunk ("a","b","c") gives [ab_id, c_id]."""

    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 1}, vocab)
    state.replay_merges([(0, 1, 3)])  # (a_id, b_id, merged_id)

    assert state.get_chunk_tokens(0) == [3, 2]


def test_bpe_state_replay_multi_merge_in_order():
    """Replaying (a,b)→ab then (ab,c)→abc collapses to a single token."""

    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 1}, vocab)
    state.replay_merges([(0, 1, 3), (3, 2, 4)])

    assert state.get_chunk_tokens(0) == [4]


def test_bpe_state_replay_left_to_right_overlap():
    """Replaying (a,a)→aa on ("a","a","a") gives [aa, a], not [a, aa]."""

    vocab = {"a": 0, "aa": 1}
    state = BpeEngine()
    state.feed({("a", "a", "a"): 1}, vocab)
    state.replay_merges([(0, 0, 1)])

    assert state.get_chunk_tokens(0) == [1, 0]


def test_bpe_state_replay_parallel_matches_serial():
    """OMP_NUM_THREADS=1 vs default must produce identical final state."""
    from src.data.bpe import _byte_encode

    vocab = {
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
            vocab.setdefault(ch, len(vocab))
        chunks[toks] = chunks.get(toks, 0) + 1

    # Pick a merge that will fire frequently: (' ', 's') if both present.
    space_id = vocab.get("Ġ")
    s_id = vocab.get("s")
    if space_id is None or s_id is None:
        pytest.skip("expected byte-encoded space and 's' in vocab")
    merged_str = "Ġs"
    merged_id = len(vocab)
    vocab[merged_str] = merged_id

    s1 = BpeEngine()
    s1.feed(chunks, vocab)
    s1.set_num_threads(1)
    s1.replay_merges([(space_id, s_id, merged_id)])

    s2 = BpeEngine()
    s2.feed(chunks, vocab)
    s2.set_num_threads(4)
    s2.replay_merges([(space_id, s_id, merged_id)])

    n = s1.get_num_chunks()
    assert n == s2.get_num_chunks()
    for i in range(n):
        assert s1.get_chunk_tokens(i) == s2.get_chunk_tokens(i)


def test_bpe_state_apply_merge_rewrites_symbols():
    """All (a,b) occurrences collapsed to merged_id."""

    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 1, ("a", "b"): 1}, vocab)
    state.apply_merge(0, 1, 3)  # merge (a,b) → ab

    # Sorted chunks: ("a","b") < ("a","b","c") → chunk 0 = [a,b], chunk 1 = [a,b,c].
    assert state.get_chunk_tokens(0) == [3]
    assert state.get_chunk_tokens(1) == [3, 2]


def test_bpe_state_apply_merge_returns_pair_deltas():
    """Returns list[(a, b, dv)] of pair-count changes for the Python heap."""

    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 1}, vocab)
    deltas = state.apply_merge(0, 1, 3)
    by_pair = {(a, b): d for a, b, d in deltas}

    # Before: (a,b):1, (b,c):1.
    # After  merge ab: (b,c) goes to 0 (delta -1), (ab,c) gains 1 (delta +1).
    # (a,b) itself is removed from internal state — not reported in deltas.
    assert by_pair == {(1, 2): -1, (3, 2): 1}


def test_bpe_state_apply_merge_overlapping_pairs_left_to_right():
    """'a a a' merging (a,a)→aa yields [aa, a] (left consume first)."""

    vocab = {"a": 0, "aa": 1}
    state = BpeEngine()
    state.feed({("a", "a", "a"): 1}, vocab)
    state.apply_merge(0, 0, 1)

    assert state.get_chunk_tokens(0) == [1, 0]
    # New pair (aa, a) gained 1.
    assert state.get_pair_count(1, 0) == 1


def test_bpe_state_apply_merge_respects_chunk_count():
    """A merge in a weight-5 chunk emits weight-5 deltas."""

    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 5}, vocab)
    deltas = state.apply_merge(0, 1, 3)
    by_pair = {(a, b): d for a, b, d in deltas}

    assert by_pair == {(1, 2): -5, (3, 2): 5}


def test_bpe_state_apply_merge_removes_pair_from_where():
    """After applying (a,b), get_chunks_by_pair(a,b) is empty."""

    vocab = {"a": 0, "b": 1, "ab": 2}
    state = BpeEngine()
    state.feed({("a", "b"): 1}, vocab)
    state.apply_merge(0, 1, 2)

    assert state.get_chunks_by_pair(0, 1) == []
    assert state.get_pair_count(0, 1) == 0


def test_bpe_state_apply_merge_dedupes_chunks_by_pair_within_chunk():
    """Repeated bigram patterns within a single chunk must not duplicate
    chunk_id entries in pairs_[...].chunk_ids for the new neighbor pairs
    created by the merge. (The chunk_ids set dedupes by construction; this
    test guards against regressions if it were ever switched back to a
    vector.)
    """

    # Chunk = [c, a, b, c, a, b]: merging (a,b)→ab creates the new pair (c, ab)
    # at TWO positions within this one chunk.
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3}
    state = BpeEngine()
    state.feed({("c", "a", "b", "c", "a", "b"): 1}, vocab)
    state.apply_merge(0, 1, 3)

    # (c, ab) should list chunk 0 EXACTLY ONCE, despite two creation positions.
    assert state.get_chunks_by_pair(2, 3) == [0]
    # And the count should still reflect both occurrences (weighted).
    assert state.get_pair_count(2, 3) == 2


def test_bpe_state_drop_pair_removes_from_chunks_by_pair_and_counts():
    """drop_pair removes the pair from internal state so it's never selected."""

    vocab = {"a": 0, "b": 1, "c": 2}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 1}, vocab)
    assert state.get_pair_count(0, 1) == 1
    assert state.get_chunks_by_pair(0, 1) != []

    state.drop_pair(0, 1)

    assert state.get_pair_count(0, 1) == 0
    assert state.get_chunks_by_pair(0, 1) == []


def test_bpe_state_feed_records_id2token():
    """feed() must populate id2token_ + token2id_ so the C++ side can build
    merged-token strings without bouncing through Python."""

    vocab = {"a": 0, "b": 1, "c": 2}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 1}, vocab)

    # New test accessors expose the native vocab to verify population.
    assert state.id2token(0) == "a"
    assert state.id2token(1) == "b"
    assert state.id2token(2) == "c"
    assert state.token2id("a") == 0
    assert state.token2id("b") == 1
    assert state.token2id("c") == 2
    assert state.get_vocab_size() == 3


def test_bpe_state_train_basic_grows_vocab():
    """train grows the native vocab from seed-size to the target."""

    vocab = {"a": 0, "b": 1, "c": 2}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 5, ("a", "b"): 3}, vocab)

    n_accepted = state.train(target_vocab_size=5)

    # Started at 3, target 5 → up to 2 accepted merges (assuming they all
    # find positive-count pairs).
    assert state.get_vocab_size() == 5
    assert n_accepted == 2
    # First accepted merge must be (a, b) — it has the highest count (8).
    merges = state.get_merges()
    assert merges[0] == ("a", "b")


def test_bpe_state_train_tie_break_smaller_id_wins():
    """On equal pair counts, the smaller (id_a, id_b) wins (HF parity)."""

    # Three pairs at equal count. (a,b)=(0,1), (a,c)=(0,2), (a,d)=(0,3).
    # Tie-break picks (a,b) first → merge "ab" gets ID 4.
    vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
    state = BpeEngine()
    state.feed(
        {("a", "b"): 3, ("a", "c"): 3, ("a", "d"): 3},
        vocab,
    )
    state.train(target_vocab_size=5)

    merges = state.get_merges()
    assert merges[0] == ("a", "b")


def test_bpe_state_train_merge_filter_vetoes():
    """A Python merge_filter callable receives (a_sym, b_sym, merged_sym)
    and can return False to veto a candidate merge."""

    def reject_ab(a: str, b: str, merged: str) -> bool:
        return merged != "ab"

    # Two competing pairs: (a,b)=count 5 (will be vetoed), (a,c)=count 3.
    # Without the filter, train would pick (a,b) first.
    vocab = {"a": 0, "b": 1, "c": 2}
    state = BpeEngine()
    state.feed({("a", "b"): 5, ("a", "c"): 3}, vocab)
    state.train(target_vocab_size=4, merge_filter=reject_ab)

    merges = state.get_merges()
    # Only (a,c) → "ac" was accepted.
    assert merges == [("a", "c")]


def test_bpe_state_get_vocab_returns_strings_and_ids():
    """get_vocab returns dict[str, int] containing all seed + merged tokens."""

    vocab = {"a": 0, "b": 1}
    state = BpeEngine()
    state.feed({("a", "b"): 2}, vocab)
    state.train(target_vocab_size=3)

    vocab = state.get_vocab()
    assert isinstance(vocab, dict)
    assert vocab == {"a": 0, "b": 1, "ab": 2}


def test_bpe_state_progress_callback_invoked_at_intervals():
    """progress_callback is called every progress_every merges with
    (vocab_size, vocab_snapshot, merges_snapshot)."""

    calls = []

    def cb(size, vocab, merges):
        calls.append((size, dict(vocab), list(merges)))

    vocab = {"a": 0, "b": 1, "c": 2}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 5}, vocab)
    state.train(
        target_vocab_size=5,
        progress_callback=cb,
        progress_every=1,  # call after every accepted merge
    )

    # 2 accepted merges → 2 callbacks.
    assert len(calls) == 2
    # Second snapshot must include both merges.
    assert len(calls[-1][2]) == 2


def test_bpe_state_progress_callback_fires_on_vocab_size_multiples():
    """progress_callback fires when get_vocab_size() (not n_accepted)
    hits a multiple of progress_every — matches Python's len(vocab) cadence."""

    calls = []

    def cb(size, vocab, merges):
        calls.append(size)

    # Seed vocab size = 3. progress_every = 4. The first callback should fire
    # when get_vocab_size() reaches 4 (after 1 accepted merge), NOT
    # when n_accepted reaches 4.
    vocab = {"a": 0, "b": 1, "c": 2}
    state = BpeEngine()
    state.feed({("a", "b", "c"): 5, ("a", "b"): 3}, vocab)
    state.train(
        target_vocab_size=5,
        progress_callback=cb,
        progress_every=4,
    )

    # Started at 3, ran to 5 → 2 accepted merges (vocab grew 3 → 4 → 5).
    # With progress_every=4, the callback fires once (when vocab hits 4),
    # not at all under n_accepted % 4 == 0 semantics (n_accepted maxes at 2).
    assert calls == [4]
