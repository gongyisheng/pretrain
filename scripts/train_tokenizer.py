"""Train a custom BPE or SuperBPE tokenizer on a HuggingFace dataset."""

import argparse
import sys

sys.path.insert(0, ".")

from datasets import load_dataset

from src.data.tokenizer_trainer import TokenizerTrainer
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config.data

    print(f"Loading dataset: {data.dataset}")
    ds = load_dataset(data.dataset, split="train", streaming=True)

    num_samples = config.tokenizer_training.num_samples

    def text_iter():
        for i, sample in enumerate(ds):
            if i >= num_samples:
                break
            yield sample["text"]

    TokenizerTrainer(config, wandb_enabled=not args.no_wandb).train(text_iter)


if __name__ == "__main__":
    main()
