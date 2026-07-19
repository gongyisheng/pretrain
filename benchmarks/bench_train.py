"""
Training throughput benchmark for GPT-2, Qwen3, and Qwen3 MoE.

Runs the Trainer for a short number of steps with W&B disabled.
Registers an on_log hook to capture per-step throughput,
excluding warmup steps (torch.compile tracing, CUDA caching, etc.).

Usage:
    # Benchmark GPT-2 124M
    python benchmarks/bench_train.py --config configs/gpt2_124m.yaml

    # Benchmark Qwen3 51M
    python benchmarks/bench_train.py --config configs/qwen3_51m.yaml

    # Benchmark Qwen3 MoE 183M (51M active)
    python benchmarks/bench_train.py --config configs/qwen3_183m_a51m.yaml

    # Run eager, no torch.compile (for A/B against the compiled run)
    python benchmarks/bench_train.py --config configs/gpt2_124m.yaml --disable-torch-compile
"""

import argparse
import json
import os
import sys

# torch reads TORCH_COMPILE_DISABLE when its dynamo module is first imported, so
# this must be set before `import torch` — a later os.environ assignment is
# ignored. Pre-scan argv so the --disable-torch-compile flag takes effect here.
if "--disable-torch-compile" in sys.argv:
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch

sys.path.insert(0, ".")

from src.utils.config import load_config
from src.training.trainer import Trainer


def run_benchmark(config_path, steps=10, warmup=5):
    """Run a training throughput benchmark using the Trainer.

    Runs all steps in a single train() call. Registers an on_log hook
    to capture throughput, excluding warmup steps.
    """
    total_steps = warmup + steps
    overrides = [
        f"training.early_stop={total_steps}",
        # disable eval/checkpoint during benchmark
        f"training.eval_every={total_steps + 1}",
        f"training.checkpoint_every={total_steps + 1}",
        "logging.log_every=1",
    ]

    config = load_config(config_path, overrides=overrides)
    trainer = Trainer(config, wandb_enabled=False)

    perf_metrics = []
    trainer.logger.register_on_log_hook(
        lambda step, metrics: perf_metrics.append(
            {
                "step": step,
                "tokens_per_sec": metrics["perf/tokens_per_sec"],
                "total_tokens": metrics["train/total_tokens"],
            }
        )
    )

    trainer.train()

    # Average tokens_per_sec from measured steps (warmup excluded)
    measured = [m for m in perf_metrics if m["step"] > warmup]
    tok_per_sec = sum(m["tokens_per_sec"] for m in measured) / len(measured)
    measured_tokens = steps * trainer.tokens_per_step
    elapsed = measured_tokens / tok_per_sec if tok_per_sec > 0 else 0

    results = {
        "config": config_path,
        "model": f"{config.model.attn_cls}+{config.model.mlp_cls}",
        "params_M": round(sum(p.numel() for p in trainer.model.parameters()) / 1e6, 1),
        "batch_size": config.training.batch_size,
        "grad_accum": config.training.gradient_accumulation_steps,
        "seq_len": config.max_seq_len,
        "tokens_per_step": trainer.tokens_per_step,
        "mixed_precision": config.training.mixed_precision,
        "activation_checkpointing": config.training.activation_checkpointing,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "elapsed_sec": round(elapsed, 2),
        "measured_tokens": measured_tokens,
        "tok_per_sec": round(tok_per_sec),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_compile": os.environ.get("TORCH_COMPILE_DISABLE") != "1",
    }

    print(f"\n{'=' * 60}")
    print(f"  {results['model']} | {results['params_M']}M")
    print(f"  GPU: {results['gpu']}")
    print(f"  Compile: {'on' if results['torch_compile'] else 'off (eager)'}")
    print(
        f"  {results['measured_steps']} steps in {results['elapsed_sec']:.1f}s (after {warmup} warmup)"
    )
    print(f"  Tokens/sec: {results['tok_per_sec']:,}")
    print(f"{'=' * 60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Training throughput benchmark")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument(
        "--steps", type=int, default=10, help="Measured steps to run (default: 10)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup steps excluded from measurement (default: 5)",
    )
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        help="Disable torch.compile (run eager). Handled before torch import via argv scan.",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save results JSON to this path"
    )
    args = parser.parse_args()

    if not args.config:
        parser.error("--config is required")

    results = run_benchmark(args.config, steps=args.steps, warmup=args.warmup)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
