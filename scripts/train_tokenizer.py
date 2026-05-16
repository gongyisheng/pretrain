"""Train a custom BPE or SuperBPE tokenizer on a HuggingFace dataset."""
import argparse
import sys

sys.path.insert(0, ".")

from datasets import load_dataset

from src.data.tokenizer import train_tokenizer
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1_000_000,
        help="Number of text samples to train on",
    )
    parser.add_argument(
        "--method",
        choices=["bpe", "superbpe"],
        default=None,
        help="Override config.data.tokenizer_method",
    )
    parser.add_argument(
        "--transition_size",
        type=int,
        default=None,
        help="Override config.data.tokenizer_transition_size (superbpe only)",
    )
    parser.add_argument(
        "--max_superword_words",
        type=int,
        default=None,
        help="Override config.data.tokenizer_max_superword_words",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Override config.data.tokenizer_path",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="superbpe")
    parser.add_argument("--wandb_eval_every", type=int, default=5000)
    parser.add_argument("--eval_num_docs", type=int, default=1000)
    args = parser.parse_args()

    config = load_config(args.config)

    method = args.method if args.method is not None else config.data.tokenizer_method
    transition_size = (
        args.transition_size
        if args.transition_size is not None
        else config.data.tokenizer_transition_size
    )
    max_superword_words = (
        args.max_superword_words
        if args.max_superword_words is not None
        else config.data.tokenizer_max_superword_words
    )
    save_path = (
        args.save_path if args.save_path is not None else config.data.tokenizer_path
    )

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
        save_path=save_path,
        method=method,
        transition_size=transition_size,
        max_superword_words=max_superword_words,
        wandb_enabled=(not args.no_wandb) and method == "superbpe",
        wandb_project=args.wandb_project,
        wandb_eval_every=args.wandb_eval_every,
        eval_num_docs=args.eval_num_docs,
    )


if __name__ == "__main__":
    main()
