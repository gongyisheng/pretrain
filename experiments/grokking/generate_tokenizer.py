"""Build the grokking tokenizer (residues 0..99 + 2 operators + EOT) and save it.

Run once before generate_data.py. Idempotent — re-running overwrites the file.
"""

import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit


def build_grokking_tokenizer() -> Tokenizer:
    """Return a WordLevel tokenizer with the fixed grokking vocab (103 tokens).

    IDs 0..99   → residue tokens "0".."99"
    IDs 100..101 → operators "o" (generic binary op) and "="
    ID 102      → end-of-text token "<|endoftext|>" (sample boundary)

    Each grokking experiment trains on a single operation (add/sub/mul/div).
    Since the model only ever sees one op, the symbol carries no information
    and is collapsed to a single token "o" — the residues + position carry
    everything the model needs to learn the table.
    """
    vocab = {str(i): i for i in range(100)}
    operators = ["o", "="]
    for i, op in enumerate(operators):
        vocab[op] = 100 + i
    vocab["<|endoftext|>"] = 102
    tokenizer = Tokenizer(WordLevel(vocab, unk_token=None))
    # WhitespaceSplit (not Whitespace) so `<|endoftext|>` isn't broken on punctuation.
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="tokenizers/grokking",
        help="Directory to write tokenizer.json into.",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = build_grokking_tokenizer()
    out_path = os.path.join(args.out_dir, "tokenizer.json")
    tokenizer.save(out_path)
    print(f"[generate_tokenizer] wrote {out_path} (vocab size = {tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
