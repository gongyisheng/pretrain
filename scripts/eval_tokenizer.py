"""Evaluate a trained tokenizer's encoding efficiency (bytes per token)."""

import argparse
import json
import sys

sys.path.insert(0, ".")

from datasets import load_dataset

from src.eval.tokenizer import evaluate


def _stream_texts(dataset: str, split: str, num_samples: int):
    ds = load_dataset(dataset, split=split, streaming=True)
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        yield sample["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to tokenizer directory (containing tokenizer.json)",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Optional path to a second tokenizer for head-to-head ratio",
    )
    parser.add_argument(
        "--dataset",
        default="openwebtext"
    )
    parser.add_argument(
        "--split",
        default="train"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000
    )
    args = parser.parse_args()

    # Note: confounders documented in
    # docs/superpowers/specs/2026-05-16-superbpe-design.md (Risk #6) —
    # the stage-1 pretokenizer for SuperBPE differs from the existing
    # custom_bpe_* tokenizers. For fair comparison, use experiments/superbpe/bpe_200k.

    print(
        f"Evaluating {args.tokenizer} on {args.dataset}:{args.split} ({args.num_samples} docs)"
    )
    result_a = evaluate(
        args.tokenizer, _stream_texts(args.dataset, args.split, args.num_samples)
    )
    print(json.dumps({"tokenizer": args.tokenizer, **result_a}, indent=2))

    if args.compare:
        print(f"Evaluating {args.compare} on the same stream")
        result_b = evaluate(
            args.compare, _stream_texts(args.dataset, args.split, args.num_samples)
        )
        print(json.dumps({"tokenizer": args.compare, **result_b}, indent=2))
        ratio = result_a["bytes_per_token"] / result_b["bytes_per_token"]
        print(f"\nRatio (A / B) = {ratio:.4f}")
        print("  > 1.0 means A is more efficient (fewer tokens per byte).")


if __name__ == "__main__":
    main()
