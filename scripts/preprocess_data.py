"""Tokenize a HuggingFace dataset and save as memory-mapped .bin files.

Uses a two-pass approach to avoid loading all tokens into RAM:
  Pass 1: Stream dataset, tokenize, write to a single temporary .bin file
  Pass 2: Split the temp file into train.bin and val.bin
"""
import argparse
import os
import sys
import numpy as np
sys.path.insert(0, ".")

from datasets import load_dataset
from src.utils.config import load_config
from src.data.tokenizer import load_tokenizer

CHUNK_SIZE = 1024 * 1024  # write in chunks of ~1M tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (None = all)")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    eot_token = tokenizer.token_to_id("<|endoftext|>")

    os.makedirs(config.data.data_dir, exist_ok=True)
    tmp_path = os.path.join(config.data.data_dir, "all_tokens.bin")
    train_path = os.path.join(config.data.data_dir, "train.bin")
    val_path = os.path.join(config.data.data_dir, "val.bin")

    # Pass 1: Stream, tokenize, and write chunks to temp file
    print(f"Loading dataset: {config.data.dataset} (streaming)")
    ds = load_dataset(config.data.dataset, split="train", streaming=True)

    total_tokens = 0
    buffer = []

    with open(tmp_path, "wb") as f:
        for i, sample in enumerate(ds):
            if args.max_samples and i >= args.max_samples:
                break
            ids = tokenizer.encode(sample["text"]).ids
            buffer.extend(ids)
            buffer.append(eot_token)

            if len(buffer) >= CHUNK_SIZE:
                chunk = np.array(buffer, dtype=np.uint16)
                f.write(chunk.tobytes())
                total_tokens += len(buffer)
                buffer = []

            if (i + 1) % 10000 == 0:
                print(f"  Tokenized {i+1} documents ({total_tokens + len(buffer):,} tokens)")

        if buffer:
            chunk = np.array(buffer, dtype=np.uint16)
            f.write(chunk.tobytes())
            total_tokens += len(buffer)

    print(f"Total tokens: {total_tokens:,}")

    # Pass 2: Split into train/val via memmap
    all_data = np.memmap(tmp_path, dtype=np.uint16, mode="r")
    n_val = int(len(all_data) * config.data.val_split)
    n_train = len(all_data) - n_val

    train_data = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(n_train,))
    train_data[:] = all_data[:n_train]
    train_data.flush()

    val_data = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(n_val,))
    val_data[:] = all_data[n_train:]
    val_data.flush()

    del all_data
    os.remove(tmp_path)

    print(f"Train: {n_train:,} tokens -> {train_path}")
    print(f"Val:   {n_val:,} tokens -> {val_path}")

    verify = np.memmap(train_path, dtype=np.uint16, mode="r")
    sample_ids = verify[:100].tolist()
    decoded = tokenizer.decode(sample_ids)
    print(f"\nSample (first 100 tokens decoded):\n{decoded[:500]}")


if __name__ == "__main__":
    main()
