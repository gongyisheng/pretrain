"""Benchmark MXFP8 E4M3 quantization for Qwen3-51M tensor shapes."""

from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.quantize._common import run_benchmark
from src.kernel.ops import quantize_mxfp8


if __name__ == "__main__":
    run_benchmark(
        "mxfp8",
        partial(quantize_mxfp8, fmt="fp8_e4m3"),
        ("blockwise1d", "blockwise2d"),
        (16, 32, 64, 128),
        (8, 9),
    )
