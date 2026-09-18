"""Benchmark NVFP4 E2M1 quantization for Qwen3-51M tensor shapes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.quantize._common import run_benchmark
from src.kernel.ops import quantize_nvfp4


if __name__ == "__main__":
    run_benchmark(
        "nvfp4",
        quantize_nvfp4,
        ("blockwise1d", "blockwise2d"),
        (16,),
        (10, 0),
        (12, 99),
    )
