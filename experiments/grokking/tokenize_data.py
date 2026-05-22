"""Tokenize a text SFT Parquet into a token-id Parquet.

Stage 2 of the SFT data pipeline. Reads ``{train,val}_text.parquet`` (columns
``question`` and ``answer``, both strings) and writes ``{train,val}.parquet``
with the same number of rows but tokenized columns:

    question_ids: list<int> — token IDs of the question text
    answer_ids:   list<int> — token IDs of the answer text

Per-sample lengths are independent and recoverable from each row.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def _tokenize_column(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    """Encode a column of strings to a column of token-id lists."""
    encs = tokenizer.encode_batch(texts)
    return [enc.ids for enc in encs]


def _tokenize_file(tokenizer: Tokenizer, in_path: str, out_path: str) -> int:
    table = pq.read_table(in_path)
    questions = table.column("question").to_pylist()
    answers = table.column("answer").to_pylist()
    q_ids = _tokenize_column(tokenizer, questions)
    a_ids = _tokenize_column(tokenizer, answers)
    out_table = pa.table({"question_ids": q_ids, "answer_ids": a_ids})
    pq.write_table(out_table, out_path)
    return len(q_ids)


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

    for split in ("train", "val"):
        in_path = os.path.join(args.data_dir, f"{split}_text.parquet")
        out_path = os.path.join(args.data_dir, f"{split}.parquet")
        n = _tokenize_file(tokenizer, in_path, out_path)
        print(f"[tokenize_data] {split}: {n} rows → {out_path}")


if __name__ == "__main__":
    main()
