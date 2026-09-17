"""Benchmark NVFP4 E2M1 quantization for Qwen3-51M tensor shapes."""

from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from benchmarks.quantize._common import run_benchmark
from src.kernel.ops import quantize_nvfp4


if __name__ == "__main__":
    run_benchmark(
        "nvfp4",
        partial(
            quantize_nvfp4,
            enable_global_scale=True,
            scale_dtype=torch.float8_e4m3fn,
            qmax=6.0,
        ),
        ("blockwise1d", "blockwise2d"),
        (16,),
        (10, 0),
        (12, 99),
    )
