"""Tokenize a text SFT Parquet into flat token + target-mask .bin files.

Stage 2 of the SFT data pipeline. Reads ``{train,val}_text.parquet`` (columns
``question`` and ``answer``, both strings) and writes two parallel flat files
per split:

    {train,val}.bin           # uint16 (or uint32) tokens, concatenated
                              # [q0..., a0..., EOT, q1..., a1..., EOT, ...]
    {train,val}_targets.bin   # uint8 mask, 1 where the token is an answer or
                              # EOT (a supervised target), 0 where it's a
                              # question token

The two files together encode the SFT structure: SFTDataset memmaps them and
applies ``labels[~targets[1:]] = -100`` to mask question-position labels.
"""

import argparse
import os

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def _tokenize_split(
    tokenizer: Tokenizer,
    in_path: str,
    out_bin: str,
    out_targets: str,
    eot_token_id: int,
    dtype: np.dtype,
) -> tuple[int, int]:
    """Tokenize one split. Returns (n_rows, n_tokens)."""
    table = pq.read_table(in_path)
    questions = table.column("question").to_pylist()
    answers = table.column("answer").to_pylist()
    q_encs = tokenizer.encode_batch(questions)
    a_encs = tokenizer.encode_batch(answers)

    tokens: list[int] = []
    targets: list[int] = []
    for q, a in zip(q_encs, a_encs):
        tokens.extend(q.ids)
        targets.extend([0] * len(q.ids))
        tokens.extend(a.ids)
        targets.extend([1] * len(a.ids))
        # EOT terminates each sample (still required as a boundary marker for
        # SFTDataset packing=False), but it is *not* a supervised target — its
        # prediction is trivial and dilutes the grokking loss/accuracy curves.
        tokens.append(eot_token_id)
        targets.append(0)

    np.asarray(tokens, dtype=dtype).tofile(out_bin)
    np.asarray(targets, dtype=np.uint8).tofile(out_targets)
    return len(q_encs), len(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing train_text.parquet and val_text.parquet.",
    )
    parser.add_argument(
        "--tokenizer_path",
        required=True,
        help="Directory containing tokenizer.json.",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(
        os.path.join(args.tokenizer_path, "tokenizer.json")
    )
    eot_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_token_id is None:
        raise ValueError(
            "tokenizer is missing the '<|endoftext|>' special token; SFTDataset "
            "needs it to delimit samples in the flat stream"
        )
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint32 if vocab_size > 65535 else np.uint16

    for split in ("train", "val"):
        in_path = os.path.join(args.data_dir, f"{split}_text.parquet")
        out_bin = os.path.join(args.data_dir, f"{split}.bin")
        out_targets = os.path.join(args.data_dir, f"{split}_targets.bin")
        n_rows, n_tokens = _tokenize_split(
            tokenizer, in_path, out_bin, out_targets, eot_token_id, dtype
        )
        print(
            f"[tokenize_data] {split}: {n_rows} rows / {n_tokens} tokens → "
            f"{out_bin}, {out_targets}"
        )


if __name__ == "__main__":
    main()
