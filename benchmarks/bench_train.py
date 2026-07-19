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

    # Model eager (flex_attention + loss still compiled), for A/B against full compile
    python benchmarks/bench_train.py --config configs/gpt2_124m.yaml --disable-torch-compile
"""

import argparse
import contextlib
import json
import os
import sys

import torch

sys.path.insert(0, ".")

from src.utils.config import load_config
from src.training.trainer import Trainer


def run_benchmark(
    config_path,
    steps=10,
    warmup=5,
    cuda_profiler=False,
    enable_torch_compile=True,
    emit_nvtx=False,
):
    """Run a training throughput benchmark using the Trainer.

    Runs all steps in a single train() call. Registers an on_log hook
    to capture throughput, excluding warmup steps.

    enable_torch_compile=False skips only the whole-model torch.compile; ops with
    their own explicit compile (flex_attention, loss) stay compiled.

    emit_nvtx=True wraps training in torch.autograd.profiler.emit_nvtx so nsys
    tags each kernel with its aten op (forward `aten::X` vs backward `XBackward0`)
    and input shapes.
    """
    total_steps = warmup + steps
    overrides = [
        f"training.early_stop={total_steps}",
        # disable eval/checkpoint during benchmark
        f"training.eval_every={total_steps + 1}",
        f"training.checkpoint_every={total_steps + 1}",
        "logging.log_every=1",
        f"training.enable_torch_compile={enable_torch_compile}",
    ]

    config = load_config(config_path, overrides=overrides)
    trainer = Trainer(config, wandb_enabled=False)

    # With --cuda-profiler (set by profile_train.sh), bracket only the measured
    # steps with cudaProfilerStart/Stop. Paired with nsys
    # --capture-range=cudaProfilerApi, this records steady state and excludes the
    # compile/autotune warmup. The log hook fires after each step, so starting at
    # `step == warmup` captures steps warmup+1..total_steps (the averaged region).
    started = stopped = False

    perf_metrics = []

    def on_log(step, metrics):
        nonlocal started, stopped
        perf_metrics.append(
            {
                "step": step,
                "tokens_per_sec": metrics["perf/tokens_per_sec"],
                "total_tokens": metrics["train/total_tokens"],
            }
        )
        if cuda_profiler and not started and step >= warmup:
            torch.cuda.profiler.start()
            started = True
        if cuda_profiler and started and not stopped and step >= total_steps:
            torch.cuda.profiler.stop()
            stopped = True

    trainer.logger.register_on_log_hook(on_log)

    nvtx_ctx = (
        torch.autograd.profiler.emit_nvtx(record_shapes=True)
        if emit_nvtx
        else contextlib.nullcontext()
    )
    with nvtx_ctx:
        trainer.train()

    # Safety net: stop if training exited before the stop step was logged.
    if cuda_profiler and started and not stopped:
        torch.cuda.profiler.stop()

    # Average tokens_per_sec from measured steps (warmup excluded)
    measured = [m for m in perf_metrics if m["step"] > warmup]
    tok_per_sec = sum(m["tokens_per_sec"] for m in measured) / len(measured)
    measured_tokens = steps * trainer.metrics.tokens_per_step
    elapsed = measured_tokens / tok_per_sec if tok_per_sec > 0 else 0

    results = {
        "config": config_path,
        "model": f"{config.model.attn_cls}+{config.model.mlp_cls}",
        "params_M": round(sum(p.numel() for p in trainer.model.parameters()) / 1e6, 1),
        "batch_size": config.training.batch_size,
        "grad_accum": config.training.gradient_accumulation_steps,
        "seq_len": config.max_seq_len,
        "tokens_per_step": trainer.metrics.tokens_per_step,
        "mixed_precision": config.training.mixed_precision,
        "activation_checkpointing": config.training.activation_checkpointing,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "elapsed_sec": round(elapsed, 2),
        "measured_tokens": measured_tokens,
        "tok_per_sec": round(tok_per_sec),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_compile": enable_torch_compile,
    }

    print(f"\n{'=' * 60}")
    print(f"  {results['model']} | {results['params_M']}M")
    print(f"  GPU: {results['gpu']}")
    compile_label = (
        "on" if enable_torch_compile else "model off (flex/loss still compiled)"
    )
    print(f"  Compile: {compile_label}")
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
        help="Skip the whole-model torch.compile (model runs eager). "
        "flex_attention and loss keep their own explicit compile.",
    )
    parser.add_argument(
        "--cuda-profiler",
        action="store_true",
        help="Bracket the measured steps with cudaProfilerStart/Stop (for nsys "
        "--capture-range=cudaProfilerApi), excluding compile/autotune warmup.",
    )
    parser.add_argument(
        "--emit-nvtx",
        action="store_true",
        help="Wrap training in emit_nvtx so nsys labels each kernel with its aten "
        "op (forward aten::X vs backward XBackward0) and input shapes.",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save results JSON to this path"
    )
    args = parser.parse_args()

    if not args.config:
        parser.error("--config is required")

    results = run_benchmark(
        args.config,
        steps=args.steps,
        warmup=args.warmup,
        cuda_profiler=args.cuda_profiler,
        enable_torch_compile=not args.disable_torch_compile,
        emit_nvtx=args.emit_nvtx,
    )

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
