"""Minimal training loop for nsys profiling.

Run with:
    mkdir logs/profiles
    nsys profile -o logs/profiles/gpt2 python benchmarks/trainer/nsys_train.py --config configs/gpt2_124m.yaml
    nsys profile --capture-range=cudaProfilerApi -o logs/profiles/qwen3 python benchmarks/trainer/nsys_train.py --config configs/qwen3_57m.yaml

Then analyze:
    nsys stats logs/profiles/gpt2.nsys-rep
    nsys stats --report cuda_gpu_kern_sum logs/profiles/qwen3.nsys-rep
"""

import argparse
import sys

import torch

sys.path.insert(0, ".")

from src.utils.config import load_config
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    total = args.warmup + args.steps
    overrides = [
        f"training.early_stop={args.warmup}",
        f"training.eval_every={total + 1}",
        f"training.checkpoint_every={total + 1}",
        "logging.log_every=1",
    ]
    config = load_config(args.config, overrides=overrides)
    trainer = Trainer(config, wandb_enabled=False)

    # Warmup (outside profiling window)
    trainer.train()

    # Profiled steps — nsys captures this range via NVTX or just wall time
    trainer.config.training.early_stop = total
    torch.cuda.cudart().cudaProfilerStart()
    trainer.train()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
