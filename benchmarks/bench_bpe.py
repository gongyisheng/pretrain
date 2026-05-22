"""
Wall-clock benchmark for BPE training.

Compares:
  1. HF `tokenizers.BpeTrainer`        (Rust baseline)
  2. native serial                     (Python control flow + C++ BpeState, 1 thread)
  3. native parallel                   (Python control flow + C++ BpeState, N threads)

Usage:
    python benchmarks/bench_bpe.py --corpus_path data/sample.txt
    python benchmarks/bench_bpe.py --corpus_path data/sample.txt --workers 1 4 8
    python benchmarks/bench_bpe.py --skip_hf --json_out results.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.data.bpe import BpeTrainer


def _corpus_factory(path: str, n_docs: int | None):
    """Build a zero-arg replayable iterable over docs in `path`."""

    def _iter():
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n_docs is not None and i >= n_docs:
                    break
                line = line.rstrip("\n")
                if line:
                    yield line

    return _iter


def _hash_output(vocab: dict, merges: list) -> str:
    """Stable hash of (vocab, merges) for determinism checks."""
    h = hashlib.sha256()
    for tok, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        h.update(f"{tok}\t{tid}\n".encode("utf-8"))
    for a, b in merges:
        h.update(f"{a} {b}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def run_hf(corpus, vocab_size: int) -> dict:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    t0 = time.perf_counter()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus(), trainer=trainer)
    return {
        "config": "hf",
        "workers": "n/a",
        "total_wall_seconds": time.perf_counter() - t0,
        "final_vocab_size": tok.get_vocab_size(),
        "hash": "n/a",  # HF and Python may differ in ID assignment, skip
    }


def run_native(corpus, vocab_size: int, n_workers: int, batch_size: int = 1000) -> dict:
    t0 = time.perf_counter()
    vocab, merges = BpeTrainer(
        vocab_size=vocab_size,
        n_workers=n_workers,
        batch_size=batch_size,
    ).train(corpus)
    return {
        "config": "native",
        "workers": n_workers,
        "total_wall_seconds": time.perf_counter() - t0,
        "final_vocab_size": len(vocab),
        "hash": _hash_output(vocab, merges),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_path", required=True, help="text file, one doc per line")
    ap.add_argument("--n_docs", type=int, default=None)
    ap.add_argument("--vocab_size", type=int, default=50_000)
    ap.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, max(1, (os.cpu_count() or 2) // 2)],
    )
    ap.add_argument("--skip_hf", action="store_true")
    ap.add_argument("--json_out", type=str, default=None)
    args = ap.parse_args()

    corpus = _corpus_factory(args.corpus_path, args.n_docs)

    results: list[dict] = []
    if not args.skip_hf:
        print(f"Running HF baseline (vocab_size={args.vocab_size})...")
        results.append(run_hf(corpus, args.vocab_size))
    for w in args.workers:
        print(f"Running native BpeTrainer (workers={w})...")
        results.append(run_native(corpus, args.vocab_size, w))

    # Determinism check across native runs.
    native_hashes = {r["hash"] for r in results if r["config"] == "native"}
    if len(native_hashes) > 1:
        print(
            f"\n!! NON-DETERMINISM: native runs produced different outputs: {native_hashes}"
        )
        sys.exit(1)

    print("\nResults:")
    print(f"{'config':<10} {'workers':<10} {'wall (s)':<12} {'vocab':<8} {'hash':<18}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['config']:<10} {str(r['workers']):<10} "
            f"{r['total_wall_seconds']:<12.2f} {r['final_vocab_size']:<8} {r['hash']:<18}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
