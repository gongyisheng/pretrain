"""
Training throughput benchmark for GPT-2 and Qwen3.

Runs the Trainer for a short number of steps (default 5) with W&B disabled,
and reports tokens/sec throughput.

Usage:
    # Benchmark GPT-2 124M (torch backend)
    python benchmarks/trainer/bench_train.py --config configs/gpt2_124m.yaml

    # Benchmark Qwen3 145M (triton backend)
    python benchmarks/trainer/bench_train.py --config configs/qwen3_145m.yaml --backend triton

    # Run all model/backend combinations
    python benchmarks/trainer/bench_train.py --all
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from src.utils.config import load_config
from src.training.trainer import Trainer


def run_benchmark(config_path, backend=None, steps=5, warmup=5):
    """Run a training throughput benchmark using the Trainer.

    Runs warmup steps first (excluded from measurement), then measures
    steady-state throughput over the remaining steps.
    """
    total_steps = warmup + steps
    overrides = [
        f"debug.max_steps={total_steps}",
        # disable eval/checkpoint during benchmark
        f"training.eval_every={total_steps + 1}",
        f"training.checkpoint_every={total_steps + 1}",
        "logging.log_every=1",
    ]
    if backend:
        overrides.append(f"training.backend={backend}")

    config = load_config(config_path, overrides=overrides)
    trainer = Trainer(config, wandb_enabled=False)

    # Phase 1: warmup (compilation, caching, etc.)
    print(f"\n--- warmup ({warmup} steps) ---")
    trainer.train()

    # Phase 2: measured steps
    print(f"\n--- benchmark ({steps} steps) ---")
    trainer.config.debug.max_steps = total_steps
    tokens_before = trainer.total_tokens

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    measured_tokens = trainer.total_tokens - tokens_before
    tok_per_sec = measured_tokens / elapsed if elapsed > 0 else 0

    results = {
        "config": config_path,
        "arch": config.model.arch,
        "params_M": round(sum(p.numel() for p in trainer.model.parameters()) / 1e6, 1),
        "backend": config.training.backend,
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
    }

    print(f"\n{'=' * 60}")
    print(f"  {results['arch']} | {results['params_M']}M | backend={results['backend']}")
    print(f"  GPU: {results['gpu']}")
    print(f"  {results['measured_steps']} steps in {results['elapsed_sec']:.1f}s (after {warmup} warmup)")
    print(f"  Tokens/sec: {results['tok_per_sec']:,}")
    print(f"{'=' * 60}\n")

    return results


def run_all_benchmarks(steps=5, warmup=5):
    """Run benchmarks for all model/backend combinations."""
    configs = [
        ("configs/gpt2_124m.yaml", "torch"),
        ("configs/gpt2_124m.yaml", "triton"),
        ("configs/qwen3_145m.yaml", "torch"),
        ("configs/qwen3_145m.yaml", "triton"),
    ]

    all_results = []
    for config_path, backend in configs:
        if not os.path.exists(config_path):
            print(f"Skipping {config_path} (not found)")
            continue
        results = run_benchmark(config_path, backend=backend, steps=steps, warmup=warmup)
        all_results.append(results)
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'=' * 70}")
    print(f"{'Model':<12} {'Params':>8} {'Backend':>8} {'tok/s':>12} {'elapsed':>10}")
    print(f"{'-' * 70}")
    for r in all_results:
        print(f"{r['arch']:<12} {r['params_M']:>7.1f}M {r['backend']:>8} {r['tok_per_sec']:>12,} {r['elapsed_sec']:>9.1f}s")
    print(f"{'=' * 70}")

    # Save results
    os.makedirs("logs/benchmarks", exist_ok=True)
    out_path = "logs/benchmarks/train_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Training throughput benchmark")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--backend", type=str, choices=["torch", "triton"], default=None,
                        help="Override backend (default: use config value)")
    parser.add_argument("--steps", type=int, default=5, help="Measured steps to run (default: 5)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps excluded from measurement (default: 5)")
    parser.add_argument("--all", action="store_true",
                        help="Run all model/backend combinations")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    if args.all:
        run_all_benchmarks(steps=args.steps, warmup=args.warmup)
        return

    if not args.config:
        parser.error("--config is required (or use --all)")

    results = run_benchmark(args.config, backend=args.backend, steps=args.steps, warmup=args.warmup)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
