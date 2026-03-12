import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_tokenizer(
    dataset_iter,
    vocab_size: int = 50257,
    save_path: str = "tokenizers/custom_bpe",
):
    """Train a BPE tokenizer on an iterator of text strings."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
    )

    tokenizer.train_from_iterator(dataset_iter, trainer=trainer)

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    print(f"Tokenizer saved to {save_path}/ (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a trained tokenizer from disk."""
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
