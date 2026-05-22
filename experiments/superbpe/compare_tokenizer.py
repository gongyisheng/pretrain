"""Compare two tokenizers by their vocabularies.

Reports:
  - Counts of common / A-only / B-only tokens
  - Jaccard similarity
  - A 2x2 four-quadrant chart (in_A x in_B) with percentages of |A union B|
  - Sample tokens from each region

Usage:
    uv run python experiments/superbpe/compare_tokenizer.py \
        --tokenizer_a tokenizers/experiments/superbpe_v150k_t140k_m2 \
        --tokenizer_b tokenizers/experiments/superbpe_v150k_t140k_m3
        --n_samples 200

    uv run python experiments/superbpe/compare_tokenizer.py \
        --tokenizer_a tmp/tokenizer_json/olmo2_p99_truncate_10G_20K_extend_200K_mw4_colon \
        --tokenizer_b tmp/tokenizer_json/olmo2_p99_truncate_10G_180K_extend_200K_mw4_colon \
        --n_samples 25
"""

import argparse
import os
import re

from tokenizers import Tokenizer

OUTPUT_ROOT = "tmp/tokenizer_compare"


def _escape(token: str) -> str:
    # Order matters: escape backslash first so we don't double-escape later.
    return (
        token.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "tok"


def _write_tokens_tsv(path: str, tokens: set[str]) -> None:
    sortd = sorted(tokens, key=lambda s: (len(s), s))
    with open(path, "w", encoding="utf-8") as f:
        f.write("token\n")
        for tok in sortd:
            f.write(_escape(tok) + "\n")


def load_tokenizer(path: str) -> Tokenizer:
    """Load a trained tokenizer from disk."""
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json"))


def _vocab_set(tokenizer) -> set[str]:
    return set(tokenizer.get_vocab().keys())


def _fmt(label: str, n: int, total: int) -> str:
    pct = 100.0 * n / total if total else 0.0
    return f"{label:>12}: {n:>10,d}  ({pct:6.2f}%)"


def _sample(tokens: set[str], k: int = 10) -> list[str]:
    # Stable, length-biased sampling: shortest + middle + longest.
    sortd = sorted(tokens, key=lambda s: (len(s), s))
    if len(sortd) <= k:
        return sortd
    step = max(1, len(sortd) // k)
    return sortd[::step][:k]


def _print_quadrants(name_a: str, name_b: str, common: int, a_only: int, b_only: int):
    union = common + a_only + b_only
    if union == 0:
        print("(both tokenizers have empty vocabularies)")
        return

    def pct(n):
        return f"{100.0 * n / union:5.2f}%"

    common_cell = f"common {pct(common)}"
    a_cell = f"A-only {pct(a_only)}"
    b_cell = f"B-only {pct(b_only)}"
    empty_cell = "    -    "

    col_a = max(len(common_cell), len(b_cell)) + 2
    col_b = max(len(a_cell), len(empty_cell)) + 2

    print()
    print(f"Four-Quadrant Chart (% of |A union B| = {union:,d})")
    print(f"  A = {name_a}")
    print(f"  B = {name_b}")
    print()

    header = f"{'':>10}  {'in B':^{col_a}} {'not in B':^{col_b}}"
    print(header)
    print(f"{'':>10} +{'-' * col_a}+{'-' * col_b}+")
    print(f"{'in A':>10} |{common_cell:^{col_a}}|{a_cell:^{col_b}}|")
    print(f"{'':>10} +{'-' * col_a}+{'-' * col_b}+")
    print(f"{'not in A':>10} |{b_cell:^{col_a}}|{empty_cell:^{col_b}}|")
    print(f"{'':>10} +{'-' * col_a}+{'-' * col_b}+")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer_a", required=True, help="Path to tokenizer A directory"
    )
    parser.add_argument(
        "--tokenizer_b", required=True, help="Path to tokenizer B directory"
    )
    parser.add_argument(
        "--name_a", default=None, help="Display name for A (default: basename)"
    )
    parser.add_argument(
        "--name_b", default=None, help="Display name for B (default: basename)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=25, help="Sample tokens to print per region"
    )
    args = parser.parse_args()

    name_a = args.name_a or os.path.basename(os.path.normpath(args.tokenizer_a))
    name_b = args.name_b or os.path.basename(os.path.normpath(args.tokenizer_b))

    tok_a = load_tokenizer(args.tokenizer_a)
    tok_b = load_tokenizer(args.tokenizer_b)
    vocab_a = _vocab_set(tok_a)
    vocab_b = _vocab_set(tok_b)

    common = vocab_a & vocab_b
    a_only = vocab_a - vocab_b
    b_only = vocab_b - vocab_a
    union = vocab_a | vocab_b
    jaccard = len(common) / len(union) if union else 0.0

    print("=" * 60)
    print(f"Tokenizer comparison: {name_a} vs {name_b}")
    print("=" * 60)
    print(f"  |A| = {len(vocab_a):,d}    |B| = {len(vocab_b):,d}")
    print(f"  |A union B| = {len(union):,d}    Jaccard = {jaccard:.4f}")
    print()
    print(_fmt("common", len(common), len(union)))
    print(_fmt(f"{name_a}-only", len(a_only), len(union)))
    print(_fmt(f"{name_b}-only", len(b_only), len(union)))

    _print_quadrants(name_a, name_b, len(common), len(a_only), len(b_only))

    print()
    print(f"Sample tokens (first {args.n_samples} by length, then lex):")
    for label, region in [
        ("common", common),
        (f"{name_a}-only", a_only),
        (f"{name_b}-only", b_only),
    ]:
        samples = _sample(region, args.n_samples)
        print(f"  [{label}] {samples}")

    out_dir = os.path.join(
        OUTPUT_ROOT, f"{_safe_filename(name_a)}_vs_{_safe_filename(name_b)}"
    )
    os.makedirs(out_dir, exist_ok=True)
    outputs = [
        ("common.tsv", common),
        (f"{_safe_filename(name_a)}_only.tsv", a_only),
        (f"{_safe_filename(name_b)}_only.tsv", b_only),
    ]
    print()
    print(f"Writing token lists to {out_dir}/")
    for fname, region in outputs:
        _write_tokens_tsv(os.path.join(out_dir, fname), region)
        print(f"  {fname}: {len(region):,d} tokens")


if __name__ == "__main__":
    main()
