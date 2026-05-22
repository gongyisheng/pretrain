"""Generate grokking train/val .bin files + meta.json for a (op, p, train_frac, seed).

Output layout (mirrors data/openwebtext/):
    data/grokking_<op>_p<p>_f<frac>/
        train.bin    # uint16, stride = 5 tokens per sample [a, op, b, =, c]
        val.bin      # same shape, disjoint pairs
        meta.json    # vocab + split metadata
"""

import argparse
import json
import os

import numpy as np
from tokenizers import Tokenizer

OPS = {"add", "sub", "mul", "div"}
OP_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


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
        for a in range(1, p):  # exclude 0
            for b in range(1, p):
                yield a, b, (a * b) % p
    elif op == "div":
        for a in range(p):
            for b in range(1, p):  # exclude b=0
                yield a, b, (a * _modinv(b, p)) % p
    else:
        raise ValueError(f"unknown op: {op}")


def _encode_sample(tokenizer: Tokenizer, a: int, b: int, op: str, c: int) -> list[int]:
    """Token IDs for [a, op_symbol, b, =, c]."""
    return tokenizer.encode(f"{a} {OP_SYMBOLS[op]} {b} = {c}").ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=sorted(OPS), required=True)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer_path",
        default="tokenizers/grokking",
        help="Directory containing tokenizer.json built by generate_tokenizer.py",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output dir (defaults to data/grokking_<op>_p<p>_f<frac>/)",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(os.path.join(args.tokenizer_path, "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()

    out_dir = args.out_dir or f"data/grokking_{args.op}_p{args.p}_f{args.train_frac}"
    os.makedirs(out_dir, exist_ok=True)

    # Build all valid pairs + encode each to 5 token IDs.
    pairs = list(_valid_pairs(args.op, args.p))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]

    n_train = int(len(pairs) * args.train_frac)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    def _write_bin(samples, path):
        flat = np.array(
            [tok for (a, b, c) in samples for tok in _encode_sample(tokenizer, a, b, args.op, c)],
            dtype=np.uint16,
        )
        flat.tofile(path)
        return flat

    train_flat = _write_bin(train_pairs, os.path.join(out_dir, "train.bin"))
    val_flat = _write_bin(val_pairs, os.path.join(out_dir, "val.bin"))

    meta = {
        "dataset": "grokking",
        "op": args.op,
        "p": args.p,
        "train_frac": args.train_frac,
        "seed": args.seed,
        "vocab_size": vocab_size,
        "question_len": 4,
        "answer_len": 1,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "special_tokens": {sym: tokenizer.token_to_id(sym) for sym in ["+", "-", "*", "/", "="]},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[generate_data] {args.op}: train={len(train_pairs)} val={len(val_pairs)} "
        f"({len(train_flat)} + {len(val_flat)} tokens) → {out_dir}"
    )


if __name__ == "__main__":
    main()
