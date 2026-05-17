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
from collections import Counter
from collections.abc import Callable, Iterable

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


def _build_chunks(
    corpus_iter: Callable[[], Iterable[str]],
    mode: str,
    n_workers: int = 1,
    batch_size: int = 1000,
) -> dict[tuple[str, ...], int]:
    """Pretokenize the corpus and aggregate by symbol-tuple.

    Returns dict {symbol_tuple: count}. Serial path used when n_workers<=1;
    parallel path added in a later task.
    """
    if n_workers > 1:
        raise NotImplementedError("parallel path added in Task 13")
    total: Counter = Counter()
    batch: list[str] = []
    for doc in corpus_iter():
        batch.append(doc)
        if len(batch) >= batch_size:
            total.update(_pretokenize_batch(batch, mode))
            batch = []
    if batch:
        total.update(_pretokenize_batch(batch, mode))
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


class _LexInverter:
    """Wrap a tuple of strings to invert lex comparison (lex-larger sorts first)."""

    __slots__ = ("v",)

    def __init__(self, v: tuple[str, str]) -> None:
        self.v = v

    def __lt__(self, other: "_LexInverter") -> bool:
        return self.v > other.v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _LexInverter) and self.v == other.v


def _make_heap_entry(
    pair: tuple[str, str],
    count: int,
) -> tuple:
    """Build a heap entry whose ordering matches HF tie-break:
      1. count desc
      2. pair lex desc (lex-larger pair wins on tie)

    `heapq` is a min-heap, so we negate count and wrap the pair in
    `_LexInverter` to flip lex comparison. The trailing `pair` tuple is the
    actual value the consumer reads back.
    """
    return (-count, _LexInverter(pair), pair)


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
        neg_count, _inverter, pair = heap[0]
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
) -> None:
    """Apply merge `pair → merged` everywhere, updating state incrementally.

    Visits only chunks in `where_to_update[pair]`. For each, scans the symbol
    list left-to-right and collapses adjacent (a, b) pairs. Adjusts
    `pair_counts` for the neighbors that change (decrement old, increment new)
    and pushes new heap entries.
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
                        heap, _make_heap_entry((prev, a), pair_counts[(prev, a)])
                    )
                pair_counts[(prev, merged)] += w
                where_to_update.setdefault((prev, merged), set()).add(cid)
                heapq.heappush(
                    heap, _make_heap_entry((prev, merged), pair_counts[(prev, merged)])
                )
            # Right neighbor: (b, next) → (merged, next).
            if i + 2 < len(syms):
                nxt = syms[i + 2]
                pair_counts[(b, nxt)] -= w
                if pair_counts[(b, nxt)] <= 0:
                    pair_counts.pop((b, nxt), None)
                else:
                    heapq.heappush(
                        heap, _make_heap_entry((b, nxt), pair_counts[(b, nxt)])
                    )
                pair_counts[(merged, nxt)] += w
                where_to_update.setdefault((merged, nxt), set()).add(cid)
                heapq.heappush(
                    heap, _make_heap_entry((merged, nxt), pair_counts[(merged, nxt)])
                )
            syms[i] = merged
            del syms[i + 1]
            i += 1


class BpeTrainer:
    """Train a BPE tokenizer in pure Python. Not yet implemented."""

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
        raise NotImplementedError

    def train(
        self,
        corpus_iter: Callable[[], Iterable[str]],
    ) -> tuple[dict[str, int], list[tuple[str, str]]]:
        raise NotImplementedError
