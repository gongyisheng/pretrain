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

from collections.abc import Callable, Iterable


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
