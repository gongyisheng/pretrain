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

import atexit
import heapq
import os
from collections import Counter
from collections.abc import Callable, Iterable
from multiprocessing import get_context

import regex as _re
from tqdm import tqdm

# Module-level shared worker pool. Spawned lazily on first use, resized only
# if a caller requests a different n_workers, torn down at process exit.
# Sharing one pool across `_build_chunks`, `_init_pair_state`, and
# `_apply_merge` amortizes the spawn cost over the whole training run
# instead of paying it on every helper call.
_MP_CTX = get_context("spawn")
_POOL = None
_POOL_N_WORKERS: int | None = None


def _get_pool(n_workers: int):
    """Return the module-level worker pool sized to ``n_workers``.

    Creates the pool on first call; if a subsequent call requests a different
    size, the existing pool is closed and a new one is spawned. Raises if
    ``n_workers < 1``.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    global _POOL, _POOL_N_WORKERS
    if _POOL is not None and _POOL_N_WORKERS == n_workers:
        return _POOL
    _close_pool()
    _POOL = _MP_CTX.Pool(n_workers)
    _POOL_N_WORKERS = n_workers
    return _POOL


def _close_pool() -> None:
    """Close and join the shared pool if one is alive. Idempotent."""
    global _POOL, _POOL_N_WORKERS
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None
        _POOL_N_WORKERS = None


atexit.register(_close_pool)


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
    return _pretokenize_batch(*args)


def _build_chunks(
    corpus_iter: Callable[[], Iterable[str]],
    mode: str,
    n_workers: int = 1,
    batch_size: int = 1000,
    show_progress: bool = False,
    progress_desc: str = "",
) -> dict[tuple[str, ...], int]:
    """Pretokenize the corpus and aggregate by symbol-tuple.

    Fans out over doc batches via the shared module pool; merges per-batch
    Counters on the main process.
    """
    pool = _get_pool(n_workers)
    total: Counter = Counter()
    results = pool.imap_unordered(
        _pretokenize_batch_starargs,
        _iter_batches(corpus_iter(), batch_size, mode),
        chunksize=1,
    )
    # Total batch count is unknown (corpus is a stream) — tqdm shows running
    # count and rate without an ETA.
    results = tqdm(
        results,
        desc=f"{progress_desc} pretokenize".strip(),
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for partial in results:
        total.update(partial)
    return dict(total)


def _replay_merges(
    chunk_slice: list[list[str]],
    merges: list[tuple[str, str]],
) -> list[list[str]]:
    """Apply ``merges`` in order to every chunk in ``chunk_slice``, in place.

    Each merge collapses adjacent (a, b) → a+b left-to-right within every
    chunk. Chunks are independent given a fixed merge list, so slices can be
    replayed in parallel — each worker processes the full merge list locally
    to preserve the cross-merge ordering dependency.

    Returns the mutated slice (same list object as the input).
    """
    for a, b in merges:
        merged = a + b
        for syms in chunk_slice:
            i = 0
            while i < len(syms) - 1:
                if syms[i] == a and syms[i + 1] == b:
                    syms[i] = merged
                    del syms[i + 1]
                else:
                    i += 1
    return chunk_slice


def _replay_merges_starargs(args):
    return _replay_merges(*args)


def _count_pairs(
    symbols_slice: list[list[str]],
    weights_slice: list[int],
    chunk_id_offset: int,
) -> tuple[Counter, dict[tuple[str, str], set[int]]]:
    """Count adjacent-pair occurrences in a slice of chunks.

    The slice's chunks are indexed starting at `chunk_id_offset` so the returned
    `where_to_update` references global chunk_ids.
    """
    pair_counts: Counter = Counter()
    where_to_update: dict[tuple[str, str], set[int]] = {}
    for i, syms in enumerate(symbols_slice):
        global_chunk_id = chunk_id_offset + i
        w = weights_slice[i]
        for x, y in zip(syms[:-1], syms[1:]):
            p = (x, y)
            pair_counts[p] += w
            where_to_update.setdefault(p, set()).add(global_chunk_id)
    return pair_counts, where_to_update


def _count_pairs_starargs(args):
    """Pickle-friendly wrapper that unpacks (symbols_slice, weights_slice,
    chunk_id_offset) for imap_unordered. Module-level (not nested) for mp.Pool
    pickling.
    """
    return _count_pairs(*args)


def _init_pair_state(
    chunks: dict[tuple[str, ...], int],
    merges_to_replay: list[tuple[str, str]],
    n_workers: int = 1,
    chunks_per_task: int = 1000,
    show_progress: bool = False,
    progress_desc: str = "",
) -> tuple[list[list[str]], list[int], Counter, dict[tuple[str, str], set[int]]]:
    """Initialize merge-loop state from chunked corpus.

    Returns:
      symbols_per_chunk: list[list[str]] — mutable symbol seq per chunk
      chunk_counts:      list[int]       — per-chunk frequency weight
      pair_counts:       Counter[(a, b)] — weighted count across all chunks
      where_to_update:   dict[(a, b), set[chunk_id]] — reverse index

    chunk_id is the position in `symbols_per_chunk`. Chunks are sorted by
    symbol tuple before id assignment for deterministic output.

    If `merges_to_replay` is non-empty, those merges are replayed on each
    chunk before pair counting (resume/extend path). Both phases fan out
    over chunk slices via the shared module pool.
    """
    pool = _get_pool(n_workers)

    # Sort for deterministic chunk_id assignment.
    sorted_items = sorted(chunks.items(), key=lambda kv: kv[0])
    symbols_per_chunk: list[list[str]] = [list(t) for t, _ in sorted_items]
    chunk_counts: list[int] = [c for _, c in sorted_items]

    n = len(symbols_per_chunk)
    chunk_starts = list(range(0, n, chunks_per_task))

    pair_counts: Counter = Counter()
    where_to_update: dict[tuple[str, str], set[int]] = {}

    # Phase 1 (resume only): replay prior merges on each slice.
    if merges_to_replay:
        replay_tasks = [
            (symbols_per_chunk[s : s + chunks_per_task], merges_to_replay)
            for s in chunk_starts
        ]
        new_symbols: list[list[str]] = []
        replay_iter = pool.imap(_replay_merges_starargs, replay_tasks, chunksize=1)
        replay_iter = tqdm(
            replay_iter,
            total=len(replay_tasks),
            desc=f"{progress_desc} replay merges".strip(),
            unit="task",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for slc in replay_iter:
            new_symbols.extend(slc)
        symbols_per_chunk = new_symbols

    # Phase 2: pair counting. Union local Counter/set returns.
    count_tasks = [
        (
            symbols_per_chunk[s : s + chunks_per_task],
            chunk_counts[s : s + chunks_per_task],
            s,
        )
        for s in chunk_starts
    ]
    count_iter = pool.imap_unordered(_count_pairs_starargs, count_tasks, chunksize=1)
    count_iter = tqdm(
        count_iter,
        total=len(count_tasks),
        desc=f"{progress_desc} count pairs".strip(),
        unit="task",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for local_pair_counts, local_where_to_update in count_iter:
        pair_counts.update(local_pair_counts)
        for p, chunk_ids in local_where_to_update.items():
            where_to_update.setdefault(p, set()).update(chunk_ids)

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


def _apply_merge_on_chunks(
    chunks: list[tuple[int, list[str], int]],
    a: str,
    b: str,
    merged: str,
) -> tuple[
    list[tuple[int, list[str]]],
    Counter,
    dict[tuple[str, str], set[int]],
]:
    """Apply merge (a, b) → merged to a slice of chunks (parallel worker).

    Each input tuple is (chunk_id, syms, weight). Mutates syms in place,
    accumulates neighbor-pair deltas and where_to_update additions locally,
    and returns them for the main process to fold in. Only chunks that
    actually changed are returned (membership in `affected` can be stale).
    """
    delta: Counter = Counter()
    wtu_adds: dict[tuple[str, str], set[int]] = {}
    mutated: list[tuple[int, list[str]]] = []
    for chunk_id, syms, w in chunks:
        changed = False
        i = 0
        while i < len(syms) - 1:
            if syms[i] != a or syms[i + 1] != b:
                i += 1
                continue
            changed = True
            if i > 0:
                prev = syms[i - 1]
                delta[(prev, a)] -= w
                delta[(prev, merged)] += w
                wtu_adds.setdefault((prev, merged), set()).add(chunk_id)
            if i + 2 < len(syms):
                nxt = syms[i + 2]
                delta[(b, nxt)] -= w
                delta[(merged, nxt)] += w
                wtu_adds.setdefault((merged, nxt), set()).add(chunk_id)
            syms[i] = merged
            del syms[i + 1]
            i += 1
        if changed:
            mutated.append((chunk_id, syms))
    return mutated, delta, wtu_adds


def _apply_merge_on_chunks_starargs(args):
    return _apply_merge_on_chunks(*args)


def _apply_merge(
    symbols_per_chunk: list[list[str]],
    chunk_counts: list[int],
    pair_counts: Counter,
    where_to_update: dict[tuple[str, str], set[int]],
    heap: list,
    pair: tuple[str, str],
    merged: str,
    vocab: dict[str, int],
    n_workers: int = 1,
    chunks_per_task: int = 1000,
) -> None:
    """Apply merge `pair → merged` everywhere, updating state incrementally.

    Visits only chunks in `where_to_update[pair]`. Per-chunk work fans out
    via the shared module pool; deltas are folded on the main process with
    one heap push per net-changed pair. `vocab` is needed by `_make_heap_entry`
    for the ID-based tie-break.
    """
    pool = _get_pool(n_workers)

    a, b = pair
    affected = where_to_update.pop(pair, set())
    pair_counts.pop(pair, None)

    affected_list = list(affected)
    # Right-size the task slice: cap at chunks_per_task but also shrink to
    # n_chunks // n_workers so small `affected` sets still keep every worker
    # busy instead of dumping a single oversized task on one worker.
    n_chunks = len(affected_list)
    effective_chunks_per_task = max(1, min(n_chunks // n_workers, chunks_per_task))
    tasks = [
        (
            [
                (chunk_id, symbols_per_chunk[chunk_id], chunk_counts[chunk_id])
                for chunk_id in affected_list[s : s + effective_chunks_per_task]
            ],
            a,
            b,
            merged,
        )
        for s in range(0, n_chunks, effective_chunks_per_task)
    ]
    total_delta: Counter = Counter()
    total_wtu_adds: dict[tuple[str, str], set[int]] = {}
    for mutated, delta, wtu_adds in pool.imap_unordered(
        _apply_merge_on_chunks_starargs, tasks, chunksize=1
    ):
        for chunk_id, new_syms in mutated:
            symbols_per_chunk[chunk_id] = new_syms
        total_delta.update(delta)
        for p, ids in wtu_adds.items():
            total_wtu_adds.setdefault(p, set()).update(ids)
    for p, dv in total_delta.items():
        pair_counts[p] += dv
        new_count = pair_counts[p]
        if new_count <= 0:
            pair_counts.pop(p, None)
        else:
            heapq.heappush(heap, _make_heap_entry(p, new_count, vocab))
    for p, ids in total_wtu_adds.items():
        where_to_update.setdefault(p, set()).update(ids)


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
        show_progress: bool = False,
        progress_desc: str = "Merging",
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
        self.show_progress = show_progress
        self.progress_desc = progress_desc

    def train(
        self,
        corpus_iter: Callable[[], Iterable[str]],
    ) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Train the tokenizer and return ``(vocab, merges)``.

        ``corpus_iter`` is a zero-arg callable yielding a fresh iterable of
        raw text docs. The trainer pretokenizes via ``self.pretokenizer``
        and, if ``initial_merges`` is set, replays them on each chunk to
        recover the post-resume state.
        """
        # 1. Pretokenize + dedup.
        chunks = _build_chunks(
            corpus_iter,
            self.pretokenizer,
            self.n_workers,
            self.batch_size,
            show_progress=self.show_progress,
            progress_desc=self.progress_desc,
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

        # 3. Build pair state. Replay so byte-level chunks catch up to the
        # post-resume state when initial_merges is set.
        merges_to_replay: list[tuple[str, str]] = list(merges)
        symbols, weights, pair_counts, where = _init_pair_state(
            chunks,
            merges_to_replay=merges_to_replay,
            n_workers=self.n_workers,
            chunks_per_task=self.batch_size,
            show_progress=self.show_progress,
            progress_desc=self.progress_desc,
        )

        # 4. Heapify.
        heap = [_make_heap_entry(p, c, vocab) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        # 5. Merge loop.
        n_vetoed = 0
        with tqdm(
            total=self.vocab_size,
            initial=len(vocab),
            desc=f"{self.progress_desc} apply_merge".strip(),
            disable=not self.show_progress,
            dynamic_ncols=True,
        ) as bar:
            while len(vocab) < self.vocab_size:
                pair, _cnt = _select_best_pair(heap, pair_counts)
                if pair is None:
                    break  # heap drained, no more pairs.
                a, b = pair
                merged = a + b
                if self.merge_filter is not None and not self.merge_filter(
                    a, b, merged
                ):
                    # Veto: drop the pair from the live state. A later merge of
                    # some neighbor of (a, b) will still decrement pair_counts[pair]
                    # in _apply_merge (because the chunk physically contains
                    # ...a b...), so this entry can transiently go negative. The
                    # `cur > 0` guard in _select_best_pair excludes it from
                    # selection — correctness is preserved, but the counter is no
                    # longer a faithful reflection of the symbol-sequence state.
                    pair_counts.pop(pair, None)
                    where.pop(pair, None)
                    n_vetoed += 1
                    if self.show_progress:
                        bar.set_postfix(vetoed=n_vetoed, refresh=False)
                    continue
                vocab[merged] = len(vocab)
                merges.append(pair)
                _apply_merge(
                    symbols,
                    weights,
                    pair_counts,
                    where,
                    heap,
                    pair,
                    merged,
                    vocab,
                    n_workers=self.n_workers,
                    chunks_per_task=self.batch_size,
                )
                bar.update(1)
                if (
                    self.progress_callback is not None
                    and len(vocab) % self.progress_every == 0
                ):
                    self.progress_callback(len(vocab), vocab, merges)

        return vocab, merges
