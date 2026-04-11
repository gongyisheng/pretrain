"""Pretrain an LLM."""
import argparse
import sys
sys.path.insert(0, ".")

from src.utils.config import load_config
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Pretrain an LLM")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args, remaining = parser.parse_known_args()

    # Parse CLI overrides (e.g. --optimizer.lr=3e-4)
    overrides = []
    for arg in remaining:
        if arg.startswith("--") and "=" in arg:
            overrides.append(arg.removeprefix("--"))

    try:
        config = load_config(args.config, overrides=overrides if overrides else None)
        trainer = Trainer(config, wandb_enabled=not args.no_wandb, resume_from=args.resume)
        trainer.train()
    except Exception as e:
        print(f"[train.py] Run failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
