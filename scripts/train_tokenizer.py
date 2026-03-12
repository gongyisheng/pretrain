"""Train a custom BPE tokenizer on a HuggingFace dataset."""
import argparse
import sys
sys.path.insert(0, ".")

from datasets import load_dataset
from src.utils.config import load_config
from src.data.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--num_samples", type=int, default=1_000_000,
                        help="Number of text samples to train on")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Loading dataset: {config.data.dataset}")
    ds = load_dataset(config.data.dataset, split="train", streaming=True)

    def text_iter():
        for i, sample in enumerate(ds):
            if i >= args.num_samples:
                break
            yield sample["text"]

    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=config.model.vocab_size,
        save_path=config.data.tokenizer_path,
    )


if __name__ == "__main__":
    main()
