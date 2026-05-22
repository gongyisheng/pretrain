"""Audit a trained tokenizer's vocabulary against the SuperBPE merge filter.

Mirrors the stage-2 filter from `src/data/tokenizer_trainer.py`:

    if ":Ġ" in merged:
        return False
    word_count = merged.count("Ġ") + (0 if merged.startswith("Ġ") else 1)

Reports the ":Ġ" rejection count and the word-count distribution over the
full vocab — pick your own cap by reading the histogram.

Usage:
    # Dumps per-bucket token lists as TSVs by default
    uv run python experiments/superbpe/test_merge_filter.py \
        --tokenizer_dir tokenizers/experiments/superbpe_v200k_t120k_m2

    uv run python experiments/superbpe/test_merge_filter.py \
        --tokenizer_dir tmp/tokenizer_json/olmo2_p99_truncate_10G_20K_extend_200K_mw4_colon
"""

import argparse
import os
import re
from collections import Counter, defaultdict

from tokenizers import Tokenizer


def _escape(token: str) -> str:
    return (
        token.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level mapping (byte -> visible unicode char)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_DECODER: dict[str, int] = {v: k for k, v in _bytes_to_unicode().items()}


def _decode_token(token: str) -> str:
    """Byte-level inverse of the ByteLevel pretokenizer. Returns the raw
    text the token represents (with leading-space markers Ġ -> ' ', etc.).
    Falls back to the original string for non-byte-level tokens like
    `<|endoftext|>`."""
    try:
        return bytes(_BYTE_DECODER[c] for c in token).decode("utf-8", errors="replace")
    except KeyError:
        return token


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "tok"


def _write_tokens_tsv(path: str, tokens: list[str]) -> None:
    sortd = sorted(tokens, key=lambda t: (len(t), t))
    with open(path, "w", encoding="utf-8") as f:
        f.write("token\tdecoded\n")
        for tok in sortd:
            f.write(f"{_escape(tok)}\t{_escape(_decode_token(tok))}\n")


def _write_tokens_with_words_tsv(path: str, tokens: list[str]) -> None:
    # Order: word_count asc, then length asc, then lex.
    sortd = sorted(tokens, key=lambda t: (_word_count(t), len(t), t))
    with open(path, "w", encoding="utf-8") as f:
        f.write("words\ttoken\tdecoded\n")
        for tok in sortd:
            f.write(
                f"{_word_count(tok)}\t{_escape(tok)}\t{_escape(_decode_token(tok))}\n"
            )


def _word_count(token: str) -> int:
    return token.count("Ġ") + (0 if token.startswith("Ġ") else 1)


def _sample(toks: list[str], k: int) -> list[str]:
    s = sorted(toks, key=lambda t: (len(t), t))
    if len(s) <= k:
        return s
    step = max(1, len(s) // k)
    return s[::step][:k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer_dir",
        required=True,
        help="Folder containing tokenizer.json",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of sample tokens to print per bucket",
    )
    parser.add_argument(
        "--write_tsv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-bucket token lists as TSV files (default: on; pass --no-write_tsv to disable)",
    )
    parser.add_argument(
        "--output_dir",
        default="tmp/merge_filter",
        help="Parent directory for TSV outputs (default: tmp/merge_filter)",
    )
    args = parser.parse_args()

    tok_path = os.path.join(args.tokenizer_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    tokens = list(tokenizer.get_vocab().keys())
    n_total = len(tokens)

    rejected_colon: list[str] = []
    by_words: dict[int, list[str]] = defaultdict(list)
    word_count_hist: Counter = Counter()

    for tok in tokens:
        if ":Ġ" in tok:
            rejected_colon.append(tok)
            continue
        wc = _word_count(tok)
        by_words[wc].append(tok)
        word_count_hist[wc] += 1

    name = os.path.basename(os.path.normpath(args.tokenizer_dir))
    print("=" * 60)
    print(f"Merge-filter audit: {name}")
    print("=" * 60)
    print(f"  total tokens: {n_total:,d}")
    print()

    def _fmt(label: str, n: int) -> str:
        pct = 100.0 * n / n_total if n_total else 0.0
        return f"  {label:<28}: {n:>10,d}  ({pct:6.2f}%)"

    print(_fmt("rejected (':Ġ' in token)", len(rejected_colon)))
    print()
    print("Word-count distribution (excluding ':Ġ' rejections):")
    for wc in sorted(word_count_hist):
        n = word_count_hist[wc]
        pct = 100.0 * n / n_total if n_total else 0.0
        print(f"    {wc:>2} words : {n:>10,d}  ({pct:6.2f}%)")

    print()
    print(f"Sample tokens (first {args.n_samples}, by length then lex):")
    print(f"  [rejected ':Ġ'] {_sample(rejected_colon, args.n_samples)}")
    for wc in sorted(by_words):
        print(f"  [{wc} words] {_sample(by_words[wc], args.n_samples)}")

    if args.write_tsv:
        out_dir = os.path.join(args.output_dir, _safe_filename(name))
        os.makedirs(out_dir, exist_ok=True)
        print()
        print(f"Writing token lists to {out_dir}/")
        _write_tokens_tsv(os.path.join(out_dir, "rejected_colon.tsv"), rejected_colon)
        print(f"  rejected_colon.tsv: {len(rejected_colon):,d} tokens")
        all_accepted = [t for toks in by_words.values() for t in toks]
        _write_tokens_with_words_tsv(
            os.path.join(out_dir, "tokens.tsv"), all_accepted
        )
        print(f"  tokens.tsv: {len(all_accepted):,d} tokens")


if __name__ == "__main__":
    main()
