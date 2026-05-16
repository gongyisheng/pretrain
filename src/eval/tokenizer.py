"""Tokenizer encoding-efficiency evaluation.

bytes_per_token = total UTF-8 byte length of input / total tokens produced.
Higher = more efficient. Special tokens are not added by encode() here.
"""

from typing import Iterable

from tokenizers import Tokenizer

from src.data.tokenizer import load_tokenizer


def _bytes_per_token(tokenizer: Tokenizer, texts: list[str]) -> float:
    """Compute bytes/token over `texts` using `tokenizer`. Special tokens excluded."""
    n_bytes = 0
    n_tokens = 0
    for t in texts:
        n_bytes += len(t.encode("utf-8"))
        n_tokens += len(tokenizer.encode(t, add_special_tokens=False).ids)
    if n_tokens == 0:
        raise ValueError("no tokens produced; corpus may be empty")
    return n_bytes / n_tokens


def evaluate(
    tokenizer_path: str,
    text_iter: Iterable[str],
    batch_size: int = 1000,
) -> dict:
    """Evaluate a saved tokenizer on a stream of texts.

    Returns a dict with n_docs, n_bytes, n_tokens, bytes_per_token, tokens_per_byte.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    n_docs = 0
    n_bytes = 0
    n_tokens = 0
    batch: list[str] = []
    for text in text_iter:
        batch.append(text)
        n_bytes += len(text.encode("utf-8"))
        n_docs += 1
        if len(batch) >= batch_size:
            encs = tokenizer.encode_batch(batch, add_special_tokens=False)
            n_tokens += sum(len(e.ids) for e in encs)
            batch = []
    if batch:
        encs = tokenizer.encode_batch(batch, add_special_tokens=False)
        n_tokens += sum(len(e.ids) for e in encs)
    if n_tokens == 0:
        raise ValueError("no tokens produced; corpus may be empty")
    return {
        "n_docs": n_docs,
        "n_bytes": n_bytes,
        "n_tokens": n_tokens,
        "bytes_per_token": n_bytes / n_tokens,
        "tokens_per_byte": n_tokens / n_bytes,
    }
