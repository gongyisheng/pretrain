"""Benchmark INT8 quantization for Qwen3-51M tensor shapes."""

from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.quantize._common import run_benchmark
from src.kernel.ops import quantize_int8


if __name__ == "__main__":
    run_benchmark(
        "int8",
        partial(quantize_int8, bits=8),
        ("tensorwise", "rowwise", "blockwise1d", "blockwise2d"),
        (16, 32, 64, 128),
        (0, 0),
    )
