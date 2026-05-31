"""Generate raw grokking SFT data as text Parquet files.

Stage 1 of the SFT data pipeline: emit (question, answer) string pairs for a
modular-arithmetic task. Stage 2 (tokenize_data.py) reads this Parquet and
produces the tokenized Parquet that the trainer consumes.

Output layout:
    data/grokking_<op>_p<p>_f<frac>/
        train_text.parquet   # columns: question, answer
        val_text.parquet     # same schema, disjoint pairs
        meta.json            # informational
"""

import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OPS = {"add", "sub", "mul", "div"}
# All ops use a single generic symbol "o" in the question string: each
# experiment trains on a single op, so the symbol carries no information and a
# uniform token keeps the vocab small.
OP_SYMBOL = "o"


def _modinv(a: int, p: int) -> int:
    """Modular multiplicative inverse via Fermat's little theorem (p prime)."""
    return pow(a, p - 2, p)


def _valid_pairs(op: str, p: int):
    """Yield (a, b, c) where c = op(a, b) mod p, with op-specific domain restrictions."""
    if op == "add":
        for a in range(p):
            for b in range(p):
                yield a, b, (a + b) % p
    elif op == "sub":
        for a in range(p):
            for b in range(p):
                yield a, b, (a - b) % p
    elif op == "mul":
        for a in range(1, p):
            for b in range(1, p):
                yield a, b, (a * b) % p
    elif op == "div":
        for a in range(p):
            for b in range(1, p):
                yield a, b, (a * _modinv(b, p)) % p
    else:
        raise ValueError(f"unknown op: {op}")


def _format_question(a: int, b: int, op: str) -> str:
    del op  # op selects the math (in _valid_pairs); the symbol is uniform "o"
    return f"{a} {OP_SYMBOL} {b} ="


def _format_answer(c: int) -> str:
    return str(c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=sorted(OPS), required=True)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output dir (defaults to data/grokking_<op>_p<p>_f<frac>/)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or f"data/grokking_{args.op}_p{args.p}_f{args.train_frac}"
    os.makedirs(out_dir, exist_ok=True)

    pairs = list(_valid_pairs(args.op, args.p))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]

    n_train = int(len(pairs) * args.train_frac)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    def _write_parquet(samples, path):
        questions = [_format_question(a, b, args.op) for (a, b, _) in samples]
        answers = [_format_answer(c) for (_, _, c) in samples]
        table = pa.table({"question": questions, "answer": answers})
        pq.write_table(table, path)

    _write_parquet(train_pairs, os.path.join(out_dir, "train_text.parquet"))
    _write_parquet(val_pairs, os.path.join(out_dir, "val_text.parquet"))

    meta = {
        "dataset": "grokking",
        "op": args.op,
        "p": args.p,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[generate_data] {args.op}: train={len(train_pairs)} val={len(val_pairs)} → {out_dir}"
    )


if __name__ == "__main__":
    main()
