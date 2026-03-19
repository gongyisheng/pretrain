"""Shared utilities for kernel benchmarks."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import triton.testing

DEFAULT_DTYPE = torch.bfloat16
DEFAULT_DEVICE = "cuda"
DEFAULT_QUANTILES = [0.5, 0.2, 0.8]
RESULTS_DIR = Path("benchmark/results")


def is_in_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1")


def get_benchmark_range(full_range: List, ci_range: List) -> List:
    return ci_range if is_in_ci() else full_range


def run_benchmark(
    fn: Callable, quantiles: List[float] = None,
) -> Tuple[float, float, float]:
    """Run benchmark using triton.testing.do_bench.

    Returns:
        Tuple of (median_us, max_us, min_us) in microseconds.
    """
    quantiles = quantiles or DEFAULT_QUANTILES
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


# -- Benchmarking helpers ------------------------------------------------------

def bench_one(
    n_rows: int, D: int, provider_fn: Callable,
    dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE,
) -> float:
    """Benchmark a single (n_rows, D, provider) configuration. Returns ms."""
    x = torch.randn(n_rows, D, dtype=dtype, device=device)
    w = torch.randn(D, dtype=dtype, device=device)
    return triton.testing.do_bench(lambda: provider_fn(x, w))


def bench_sweep(
    providers: Dict[str, Callable],
    sweep_param: str, sweep_vals: List[int], fixed_params: dict,
) -> Dict[str, List[float]]:
    """Benchmark all providers across a sweep variable."""
    results = {name: [] for name in providers}
    for val in sweep_vals:
        if sweep_param == "D":
            n_rows, D = fixed_params["n_rows"], val
        else:
            n_rows, D = val, fixed_params["D"]
        for name, fn in providers.items():
            ms = bench_one(n_rows, D, fn)
            results[name].append(ms)
        print(f"  {sweep_param}={val} done")
    return results


# -- Plotting helpers ----------------------------------------------------------

def plot_speedup_bars(
    ax, sweep_vals: List[int], results: Dict[str, List[float]],
    baseline: str, colors: Dict[str, str],
    xlabel: str, title: str,
):
    """Bar chart showing speedup of each implementation over baseline."""
    baseline_times = np.array(results[baseline])
    providers = [p for p in results if p != baseline]
    n_providers = len(providers)
    x = np.arange(len(sweep_vals))
    width = 0.8 / n_providers

    for i, name in enumerate(providers):
        speedups = baseline_times / np.array(results[name])
        bars = ax.bar(
            x + i * width - (n_providers - 1) * width / 2, speedups,
            width, label=name, color=colors[name], edgecolor="white", linewidth=0.5,
        )
        for bar, s in zip(bars, speedups):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{s:.1f}x", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in sweep_vals])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Speedup over {baseline}")
    ax.set_title(title)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)


def plot_latency_lines(
    ax, sweep_vals: List[int], results: Dict[str, List[float]],
    colors: Dict[str, str], xlabel: str, title: str,
):
    """Line chart showing raw latency (log-log)."""
    for name in results:
        ax.plot(
            sweep_vals, results[name], marker="o", markersize=4,
            label=name, color=colors[name], linewidth=1.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
