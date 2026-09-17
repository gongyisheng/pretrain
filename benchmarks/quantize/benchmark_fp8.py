"""Benchmark E4M3 FP8 quantization with FP32 scales on Qwen3 51M operands."""

from functools import partial
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.quantize._common import run_benchmark
from src.kernel.ops import quantize_fp8


if __name__ == "__main__":
    run_benchmark(
        "fp8",
        partial(quantize_fp8, dtype=torch.float8_e4m3fn, scale_dtype=torch.float32),
        ("tensorwise", "rowwise", "blockwise1d", "blockwise2d"),
        (16, 32, 64, 128),
        (8, 9),
    )
