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
