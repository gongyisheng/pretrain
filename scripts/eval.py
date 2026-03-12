"""Evaluate a pretrained model checkpoint."""
import argparse
import sys
import os
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")

from src.utils.config import load_config
from src.model.registry import build_model
from src.data.dataset import PretrainDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load val data
    val_path = os.path.join(config.data.data_dir, "val.bin")
    val_dataset = PretrainDataset(val_path, config.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    eval_steps = args.eval_steps or config.training.eval_steps
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= eval_steps:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Step: {ckpt.get('step', 'unknown')}")
    print(f"Val loss: {avg_loss:.4f}")
    print(f"Val perplexity: {ppl:.2f}")
    print(f"Evaluated on {n_batches} batches")


if __name__ == "__main__":
    main()
