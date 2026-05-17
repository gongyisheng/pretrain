"""Pure-Python BPE trainer.

Single public class `BpeTrainer`. Produces (vocab, merges) byte-identical to
HuggingFace `tokenizers.BpeTrainer` on the same corpus + pretokenizer, and
adds two capabilities HF lacks:

  - Resume/extend from a saved tokenizer (via `initial_vocab` + `initial_merges`).
  - Per-merge veto via `merge_filter` callback (used by SuperBPE stage 2 for
    `max_superword_words` and `:Ġ` exclusion).

Encode/decode at inference stays on HF: emit a HF-compatible (vocab, merges)
pair, caller wraps in `tokenizers.Tokenizer(models.BPE(...))` and saves.
"""

import heapq
import os
from collections import Counter
from collections.abc import Callable, Iterable
from multiprocessing import get_context

import regex as _re


def _build_byte_to_unicode() -> dict[int, str]:
    """GPT-2 byte→unicode mapping. Byte-for-byte identical to
    `tokenizers.pre_tokenizers.ByteLevel`'s mapping.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_TO_UNICODE: dict[int, str] = _build_byte_to_unicode()
_UNICODE_TO_BYTE: dict[str, int] = {c: b for b, c in _BYTE_TO_UNICODE.items()}


def _byte_encode(text: str) -> str:
    """Encode `text` as a byte-level unicode string. Each output char = one input byte."""
    return "".join(_BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))


# Paper-default whitespace pretokenization (HF / GPT-2, no contraction rule).
_PRETOK_BPE_REGEX = _re.compile(r" ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
# Digit lookahead: split runs of digits into groups of 3 from the right.
_PRETOK_DIGIT_REGEX = _re.compile(r"(?=(\d{3})+(?!\d))")


def _split_digit_groups(piece: str) -> list[str]:
    """Apply the (?=(\\d{3})+(?!\\d)) split. Equivalent to HF's Split(behavior='isolated')."""
    if not any(c.isdigit() for c in piece):
        return [piece]
    splits = [m.start() for m in _PRETOK_DIGIT_REGEX.finditer(piece)]
    if not splits:
        return [piece]
    pieces: list[str] = []
    prev = 0
    for s in splits:
        if s > prev:
            pieces.append(piece[prev:s])
        prev = s
    pieces.append(piece[prev:])
    return [p for p in pieces if p]


def _pretokenize(doc: str, mode: str) -> list[str]:
    """Return a list of byte-encoded pieces ready for BPE.

    mode="bpe":       regex split (whitespace + digits) → byte-encode each piece.
    mode="bytelevel": no word splitting; whole doc as one byte-encoded chunk.
    """
    if mode == "bytelevel":
        return [_byte_encode(doc)]
    if mode != "bpe":
        raise ValueError(
            f"unknown pretokenizer mode: {mode!r}; expected 'bpe' or 'bytelevel'"
        )
    # Simulate HF Split(behavior="isolated"): cover the full string, including
    # segments between matches (e.g., pure-digit runs not captured by the letter/
    # punctuation alternatives).
    segments: list[str] = []
    prev = 0
    for m in _PRETOK_BPE_REGEX.finditer(doc):
        start, end = m.start(), m.end()
        if start > prev:
            segments.append(doc[prev:start])  # unmatched segment (e.g., digits)
        segments.append(doc[start:end])
        prev = end
    if prev < len(doc):
        segments.append(doc[prev:])  # trailing unmatched
    pieces: list[str] = []
    for seg in segments:
        for sub in _split_digit_groups(seg):
            pieces.append(_byte_encode(sub))
    return pieces


def _pretokenize_batch(batch: list[str], mode: str) -> Counter:
    """Pretokenize a batch of documents, return Counter of symbol-tuples.

    Module-level (not nested) for `mp.Pool` pickling. Used by `_build_chunks`
    in both serial (n_workers=1) and parallel paths.
    """
    counts: Counter = Counter()
    for doc in batch:
        for piece in _pretokenize(doc, mode):
            counts[tuple(piece)] += 1
    return counts


def _iter_batches(stream, batch_size, mode):
    """Yield (batch, mode) tuples from a stream of documents."""
    batch: list[str] = []
    for doc in stream:
        batch.append(doc)
        if len(batch) >= batch_size:
            yield (batch, mode)
            batch = []
    if batch:
        yield (batch, mode)


def _pretokenize_batch_starargs(args):
    """Pickle-friendly wrapper that unpacks (batch, mode) for imap_unordered."""
    batch, mode = args
    return _pretokenize_batch(batch, mode)


def _build_chunks(
    corpus_iter: Callable[[], Iterable[str]],
    mode: str,
    n_workers: int = 1,
    batch_size: int = 1000,
) -> dict[tuple[str, ...], int]:
    """Pretokenize the corpus and aggregate by symbol-tuple.

    n_workers <= 1: serial.
    n_workers > 1: mp.Pool fan-out over doc batches; merge Counters on main.
    """
    total: Counter = Counter()
    if n_workers <= 1:
        batch: list[str] = []
        for doc in corpus_iter():
            batch.append(doc)
            if len(batch) >= batch_size:
                total.update(_pretokenize_batch(batch, mode))
                batch = []
        if batch:
            total.update(_pretokenize_batch(batch, mode))
        return dict(total)

    # Parallel: feed batches into a Pool. Use spawn context for clean state.
    ctx = get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        results = pool.imap_unordered(
            _pretokenize_batch_starargs,
            _iter_batches(corpus_iter(), batch_size, mode),
            chunksize=1,
        )
        for partial in results:
            total.update(partial)
    return dict(total)


def _init_pair_state(
    chunks: dict[tuple[str, ...], int],
    merges_to_replay: list[tuple[str, str]],
) -> tuple[list[list[str]], list[int], Counter, dict[tuple[str, str], set[int]]]:
    """Initialize merge-loop state from chunked corpus.

    Returns:
      symbols_per_chunk: list[list[str]] — mutable symbol seq per chunk
      chunk_counts:      list[int]       — per-chunk frequency weight
      pair_counts:       Counter[(a, b)] — weighted count across all chunks
      where_to_update:   dict[(a, b), set[chunk_id]] — reverse index

    chunk_id is the position in `symbols_per_chunk`. Chunks are sorted by their
    initial symbol tuple before id assignment so output is deterministic
    regardless of `chunks` dict insertion order.

    If `merges_to_replay` is non-empty, those merges are applied in order to
    each chunk's symbol list before pair counts are built — this is the
    resume/extend code path.
    """
    # Sort for deterministic chunk_id assignment.
    sorted_items = sorted(chunks.items(), key=lambda kv: kv[0])
    symbols_per_chunk: list[list[str]] = [list(t) for t, _ in sorted_items]
    chunk_counts: list[int] = [c for _, c in sorted_items]

    # Apply replay merges in order. Each merge collapses adjacent (a, b) → a+b
    # in every chunk, left-to-right.
    for a, b in merges_to_replay:
        merged = a + b
        for syms in symbols_per_chunk:
            i = 0
            while i < len(syms) - 1:
                if syms[i] == a and syms[i + 1] == b:
                    syms[i] = merged
                    del syms[i + 1]
                else:
                    i += 1

    pair_counts: Counter = Counter()
    where_to_update: dict[tuple[str, str], set[int]] = {}
    for cid, syms in enumerate(symbols_per_chunk):
        w = chunk_counts[cid]
        for x, y in zip(syms[:-1], syms[1:]):
            pair_counts[(x, y)] += w
            where_to_update.setdefault((x, y), set()).add(cid)

    return symbols_per_chunk, chunk_counts, pair_counts, where_to_update


def _make_heap_entry(
    pair: tuple[str, str],
    count: int,
    vocab: dict[str, int],
) -> tuple:
    """Build a heap entry whose ordering matches HF tie-break:
      1. count desc
      2. (vocab_id_a, vocab_id_b) asc — i.e., the pair with the smaller token
         IDs wins on a count tie, matching HF's Rust `Ord` impl which compares
         `other.pair.cmp(&self.pair)` (ascending on IDs) in a max-heap.

    `heapq` is a min-heap, so we negate count. The ID tuple sorts naturally in
    ascending order, so the smallest-ID pair bubbles to the top on ties.
    The trailing `pair` tuple is the actual value the consumer reads back.
    """
    id_key = (vocab[pair[0]], vocab[pair[1]])
    return (-count, id_key, pair)


def _select_best_pair(
    heap: list,
    pair_counts: Counter,
) -> tuple[tuple[str, str] | None, int]:
    """Pop heap until a non-stale entry is found.

    Returns (pair, count) or (None, 0) if heap drained without finding a
    live entry. Stale = heap entry's count disagrees with current
    `pair_counts[pair]`.
    """
    while heap:
        neg_count, _id_key, pair = heap[0]
        count = -neg_count
        cur = pair_counts.get(pair, 0)
        if cur == count and cur > 0:
            heapq.heappop(heap)
            return pair, cur
        heapq.heappop(heap)
    return None, 0


def _apply_merge(
    symbols_per_chunk: list[list[str]],
    chunk_counts: list[int],
    pair_counts: Counter,
    where_to_update: dict[tuple[str, str], set[int]],
    heap: list,
    pair: tuple[str, str],
    merged: str,
    vocab: dict[str, int],
) -> None:
    """Apply merge `pair → merged` everywhere, updating state incrementally.

    Visits only chunks in `where_to_update[pair]`. For each, scans the symbol
    list left-to-right and collapses adjacent (a, b) pairs. Adjusts
    `pair_counts` for the neighbors that change (decrement old, increment new)
    and pushes new heap entries. `vocab` is needed by `_make_heap_entry` for
    the ID-based tie-break.
    """
    a, b = pair
    affected = where_to_update.pop(pair, set())
    pair_counts.pop(pair, None)
    for cid in affected:
        syms = symbols_per_chunk[cid]
        w = chunk_counts[cid]
        i = 0
        while i < len(syms) - 1:
            if syms[i] != a or syms[i + 1] != b:
                i += 1
                continue
            # Left neighbor: (prev, a) → (prev, merged).
            if i > 0:
                prev = syms[i - 1]
                pair_counts[(prev, a)] -= w
                if pair_counts[(prev, a)] <= 0:
                    pair_counts.pop((prev, a), None)
                else:
                    heapq.heappush(
                        heap,
                        _make_heap_entry((prev, a), pair_counts[(prev, a)], vocab),
                    )
                pair_counts[(prev, merged)] += w
                where_to_update.setdefault((prev, merged), set()).add(cid)
                heapq.heappush(
                    heap,
                    _make_heap_entry(
                        (prev, merged), pair_counts[(prev, merged)], vocab
                    ),
                )
            # Right neighbor: (b, next) → (merged, next).
            if i + 2 < len(syms):
                nxt = syms[i + 2]
                pair_counts[(b, nxt)] -= w
                if pair_counts[(b, nxt)] <= 0:
                    pair_counts.pop((b, nxt), None)
                else:
                    heapq.heappush(
                        heap,
                        _make_heap_entry((b, nxt), pair_counts[(b, nxt)], vocab),
                    )
                pair_counts[(merged, nxt)] += w
                where_to_update.setdefault((merged, nxt), set()).add(cid)
                heapq.heappush(
                    heap,
                    _make_heap_entry((merged, nxt), pair_counts[(merged, nxt)], vocab),
                )
            syms[i] = merged
            del syms[i + 1]
            i += 1


class BpeTrainer:
    """Train a BPE tokenizer in pure Python. Produces (vocab, merges)
    byte-identical to HF's `tokenizers.BpeTrainer`.
    """

    _VALID_PRETOK_MODES = ("bpe", "bytelevel")

    def __init__(
        self,
        vocab_size: int,
        special_tokens: tuple[str, ...] = ("<|endoftext|>",),
        pretokenizer: str = "bpe",
        initial_vocab: dict[str, int] | None = None,
        initial_merges: list[tuple[str, str]] | None = None,
        merge_filter: Callable[[str, str, str], bool] | None = None,
        progress_callback: Callable[[int, dict, list], None] | None = None,
        progress_every: int = 1000,
        n_workers: int | None = None,
        batch_size: int = 1000,
    ) -> None:
        if pretokenizer not in self._VALID_PRETOK_MODES:
            raise ValueError(
                f"unknown pretokenizer mode: {pretokenizer!r}; "
                f"expected one of {self._VALID_PRETOK_MODES}"
            )
        # Minimum vocab = alphabet (256) + |specials|.
        min_vocab = 256 + len(special_tokens)
        if vocab_size < min_vocab:
            raise ValueError(
                f"vocab_size={vocab_size} too small; need at least "
                f"{min_vocab} (256 alphabet + {len(special_tokens)} specials)"
            )
        # initial_vocab and initial_merges are paired.
        if (initial_vocab is None) != (initial_merges is None):
            raise ValueError(
                "initial_vocab and initial_merges must both be provided or both None"
            )
        if initial_vocab is not None:
            if len(initial_vocab) > vocab_size:
                raise ValueError(
                    f"initial_vocab size {len(initial_vocab)} > vocab_size {vocab_size}"
                )
            if sorted(initial_vocab.values()) != list(range(len(initial_vocab))):
                raise ValueError("initial_vocab IDs must be contiguous from 0")
            for a, b in initial_merges:  # type: ignore[union-attr]
                if a not in initial_vocab:
                    raise ValueError(f"initial_merges references unknown symbol: {a!r}")
                if b not in initial_vocab:
                    raise ValueError(f"initial_merges references unknown symbol: {b!r}")
                if (a + b) not in initial_vocab:
                    raise ValueError(
                        f"initial_merges produces {a + b!r} but it is not in initial_vocab"
                    )

        self.vocab_size = vocab_size
        self.special_tokens = tuple(special_tokens)
        self.pretokenizer = pretokenizer
        self.initial_vocab = initial_vocab
        self.initial_merges = initial_merges
        self.merge_filter = merge_filter
        self.progress_callback = progress_callback
        self.progress_every = progress_every
        self.n_workers = (
            max(1, (os.cpu_count() or 2) // 2)
            if n_workers is None
            else max(1, n_workers)
        )
        self.batch_size = batch_size

    def train(
        self,
        corpus_iter: Callable[[], Iterable[str]],
    ) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Train the tokenizer and return ``(vocab, merges)``.

        ``corpus_iter`` is a zero-arg callable that yields a fresh iterable of
        documents each time it is called. The factory pattern (rather than a
        plain iterable) lets the trainer re-stream the corpus internally —
        currently used by the multi-pass dataset wrappers in
        ``src.data.tokenizer_trainer`` and required by SuperBPE stage 2.
        """
        # 1. Pretokenize + dedup.
        chunks = _build_chunks(
            corpus_iter, self.pretokenizer, self.n_workers, self.batch_size
        )
        if not chunks:
            raise ValueError("corpus produced no trainable chunks")

        # 2. Seed vocab: specials + byte alphabet (or initial_vocab if resuming).
        if self.initial_vocab is None:
            vocab: dict[str, int] = {}
            for sp in self.special_tokens:
                vocab[sp] = len(vocab)
            for c in sorted(
                _BYTE_TO_UNICODE.values()
            ):  # IDs in lex-sorted unicode order
                vocab[c] = len(vocab)
            merges: list[tuple[str, str]] = []
        else:
            vocab = dict(self.initial_vocab)
            merges = list(self.initial_merges)  # type: ignore[arg-type]

        # 3. Build pair state (replays initial_merges if any).
        symbols, weights, pair_counts, where = _init_pair_state(
            chunks, merges_to_replay=merges
        )

        # 4. Heapify.
        heap = [_make_heap_entry(p, c, vocab) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        # 5. Merge loop.
        while len(vocab) < self.vocab_size:
            pair, _cnt = _select_best_pair(heap, pair_counts)
            if pair is None:
                break  # heap drained, no more pairs.
            a, b = pair
            merged = a + b
            if self.merge_filter is not None and not self.merge_filter(a, b, merged):
                # Veto: drop the pair from the live state. A later merge of
                # some neighbor of (a, b) will still decrement pair_counts[pair]
                # in _apply_merge (because the chunk physically contains
                # ...a b...), so this entry can transiently go negative. The
                # `cur > 0` guard in _select_best_pair excludes it from
                # selection — correctness is preserved, but the counter is no
                # longer a faithful reflection of the symbol-sequence state.
                pair_counts.pop(pair, None)
                where.pop(pair, None)
                continue
            vocab[merged] = len(vocab)
            merges.append(pair)
            _apply_merge(
                symbols, weights, pair_counts, where, heap, pair, merged, vocab
            )
            if (
                self.progress_callback is not None
                and len(vocab) % self.progress_every == 0
            ):
                self.progress_callback(len(vocab), vocab, merges)

        return vocab, merges
