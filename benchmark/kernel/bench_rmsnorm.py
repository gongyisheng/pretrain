"""Benchmark: single-pass vs tiled RMSNorm kernel.

Usage:
    python -m benchmark.kernel.bench_rmsnorm
"""

import itertools

import torch
import triton
import triton.testing

from benchmark.kernel.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    bench_sweep,
    ensure_results_dir,
    get_benchmark_range,
    plot_latency_lines,
    plot_speedup_bars,
    run_benchmark,
)
from src.kernel.rmsnorm import triton_rmsnorm, triton_rmsnorm_2d, triton_rmsnorm_tiled


import torch.nn.functional as F


def torch_naive_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Hand-written PyTorch (multiple kernels, no fusion)."""
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def torch_builtin_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """torch.nn.functional.rms_norm (cuDNN-fused)."""
    return F.rms_norm(x, weight.shape, weight, eps)


@torch.compile()
def torch_compile_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """torch.compile'd naive implementation."""
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x.to(dtype)


# -- Benchmark ranges (reduced in CI) -----------------------------------------

ROWS_LIST = get_benchmark_range(
    full_range=[64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    ci_range=[2048],
)
D_LIST = get_benchmark_range(
    full_range=[128, 256, 512, 768, 1024, 2048, 4096, 8192],
    ci_range=[768, 4096],
)

MAX_ELEMENTS = 64 * 1024 * 1024  # ~128MB in bf16, safe for 12GB GPU
configs = list(itertools.product(D_LIST, ROWS_LIST))

LINE_VALS = ["triton_single", "triton_2d", "triton_tiled", "torch_builtin", "torch_compile", "torch_naive"]
LINE_NAMES = ["Triton Single-pass", "Triton 2D", "Triton Tiled", "Torch Builtin", "Torch Compile", "Torch Naive"]
STYLES = [("blue", "-"), ("cyan", "-"), ("green", "--"), ("orange", "-"), ("purple", "-."), ("red", ":")]


# -- Benchmark ----------------------------------------------------------------

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["D", "n_rows"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="rmsnorm-performance",
        args={},
    )
)
def benchmark(D: int, n_rows: int, provider: str):
    x = torch.randn(n_rows, D, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    w = torch.randn(D, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)

    fn_map = {
        "triton_single": lambda: triton_rmsnorm(x, w),
        "triton_2d": lambda: triton_rmsnorm_2d(x, w),
        "triton_tiled": lambda: triton_rmsnorm_tiled(x, w),
        "torch_builtin": lambda: torch_builtin_rmsnorm(x, w),
        "torch_compile": lambda: torch_compile_rmsnorm(x, w),
        "torch_naive": lambda: torch_naive_rmsnorm(x, w),
    }
    return run_benchmark(fn_map[provider])


def plot():
    """Generate publication-style 4-panel benchmark plot."""
    import matplotlib.pyplot as plt

    providers = {
        "Triton Single-pass": lambda x, w: triton_rmsnorm(x, w),
        "Triton 2D": lambda x, w: triton_rmsnorm_2d(x, w),
        "Triton Tiled": lambda x, w: triton_rmsnorm_tiled(x, w),
        "Torch Builtin": lambda x, w: torch_builtin_rmsnorm(x, w),
        "Torch Compile": lambda x, w: torch_compile_rmsnorm(x, w),
        "Torch Naive": lambda x, w: torch_naive_rmsnorm(x, w),
    }
    colors = {
        "Triton Single-pass": "#2563eb",
        "Triton 2D": "#06b6d4",
        "Triton Tiled": "#16a34a",
        "Torch Builtin": "#ea580c",
        "Torch Compile": "#9333ea",
        "Torch Naive": "#dc2626",
    }
    baseline = "Torch Naive"
    results_dir = ensure_results_dir()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RMSNorm Kernel Benchmark (bf16)", fontsize=14, fontweight="bold")

    print("Sweeping D (rows=2048)...")
    D_vals = [128, 256, 512, 768, 1024, 2048, 4096, 8192, 16384]
    results_d = bench_sweep(providers, "D", D_vals, {"n_rows": 2048})
    plot_speedup_bars(axes[0, 0], D_vals, results_d, baseline, colors,
                      "Hidden Dimension (D)", "Speedup vs D (n_rows=2048)")

    print("Sweeping rows (D=768)...")
    row_vals = [64, 256, 1024, 4096, 16384]
    results_rows = bench_sweep(providers, "n_rows", row_vals, {"D": 768})
    plot_speedup_bars(axes[0, 1], row_vals, results_rows, baseline, colors,
                      "Number of Rows", "Speedup vs Rows (D=768)")

    plot_latency_lines(axes[1, 0], D_vals, results_d, colors,
                       "Hidden Dimension (D)", "Latency vs D (n_rows=2048)")
    plot_latency_lines(axes[1, 1], row_vals, results_rows, colors,
                       "Number of Rows", "Latency vs Rows (D=768)")

    plt.tight_layout()
    out_path = results_dir / "rmsnorm-benchmark.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    results_dir = ensure_results_dir()
    benchmark.run(print_data=True, save_path=str(results_dir))
    plot()
