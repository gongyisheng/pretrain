"""BPE trainer with a C++ hot-loop extension.

Single public class `BpeTrainer`. Produces (vocab, merges) byte-identical to
HuggingFace `tokenizers.BpeTrainer` on the same corpus + pretokenizer, and
adds two capabilities HF lacks:

  - Resume/extend from a saved tokenizer (via `initial_vocab` + `initial_merges`).
  - Hardcoded SuperBPE filter rules toggled via `max_superword_words` and
    `forbid_colon_g` (the C++ side enforces them inside `run_merge_loop`).

Encode/decode at inference stays on HF: emit a HF-compatible (vocab, merges)
pair, caller wraps in `tokenizers.Tokenizer(models.BPE(...))` and saves.
"""

import atexit
import os
from collections import Counter
from collections.abc import Callable, Iterable
from multiprocessing import get_context

import regex as _re
from tqdm import tqdm

from src.data.bpe_native import BpeState

# Module-level shared worker pool. Spawned lazily on first use, resized only
# if a caller requests a different n_workers, torn down at process exit.
# Used by `_build_chunks` for pretokenize fan-out (the only mp-parallel
# step left after the C++ BpeState rewire). Keeping it module-level lets
# us reuse the same pool across multiple `BpeTrainer.train()` calls in one
# process (e.g., SuperBPE's subword + superword passes).
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


class BpeTrainer:
    """Train a BPE tokenizer. Python control flow drives a C++ `BpeState`
    extension for the hot loops (replay merges + per-merge chunk scan).
    Produces (vocab, merges) byte-identical to HF's `tokenizers.BpeTrainer`.
    """

    _VALID_PRETOK_MODES = ("bpe", "bytelevel")

    def __init__(
        self,
        vocab_size: int,
        special_tokens: tuple[str, ...] = ("<|endoftext|>",),
        pretokenizer: str = "bpe",
        initial_vocab: dict[str, int] | None = None,
        initial_merges: list[tuple[str, str]] | None = None,
        max_superword_words: int = -1,
        forbid_colon_g: bool = False,
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
        self.max_superword_words = max_superword_words
        self.forbid_colon_g = forbid_colon_g
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
        # 1. Pretokenize + dedup. Still mp.Pool-parallel (regex bound).
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

        # 2. Seed vocab: specials + byte alphabet (or initial_vocab on resume).
        if self.initial_vocab is None:
            vocab: dict[str, int] = {}
            for sp in self.special_tokens:
                vocab[sp] = len(vocab)
            for c in sorted(_BYTE_TO_UNICODE.values()):  # lex-sorted unicode order
                vocab[c] = len(vocab)
            merges: list[tuple[str, str]] = []
        else:
            vocab = dict(self.initial_vocab)
            merges = list(self.initial_merges)  # type: ignore[arg-type]

        # 3. Native state: seed + replay (on resume) + initial pair counts.
        state = BpeState()
        state.set_num_threads(self.n_workers)
        state.seed(chunks, vocab)
        del chunks  # large; release before pair counting allocates more.
        if merges:
            sym_to_id = vocab  # already keyed by str; dict lookup is O(1)
            merges_int = [
                (sym_to_id[a], sym_to_id[b], sym_to_id[a + b]) for a, b in merges
            ]
            state.replay_merges(merges_int)
        state.build_initial_pairs()

        # 4. Configure SuperBPE filter (both default off).
        if self.max_superword_words > 0:
            state.set_max_superword_words(self.max_superword_words)
        if self.forbid_colon_g:
            state.set_forbid_colon_g(True)

        # 5. Run the merge loop natively.
        state.run_merge_loop(
            target_vocab_size=self.vocab_size,
            progress_cb=self.progress_callback,
            progress_every=self.progress_every,
            show_progress=self.show_progress,
            progress_desc=self.progress_desc,
        )

        # 6. Marshal final output back to Python. Native `get_merges()` returns
        #    only the merges added during `run_merge_loop`; on resume, prepend
        #    the prior `initial_merges` (which were replayed onto chunks but
        #    never recorded in `merges_native_`).
        return state.get_vocab(), merges + state.get_merges()
