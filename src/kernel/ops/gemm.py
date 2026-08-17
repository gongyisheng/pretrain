import torch

import src.kernel.backends.cublaslt  # noqa: F401
import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.selector import dispatch

__all__ = [
    "grouped_mm",
    "int8_scaled_mm",
    "fp8_scaled_mm",
    "mxfp8_scaled_mm",
    "int8_scaled_grouped_mm",
    "fp8_scaled_grouped_mm",
    "mxfp8_scaled_grouped_mm",
]

# Triton and cuBLASLt are both eligible for gemm.mxfp8_scaled_mm on sm100+, so
# dispatch can't pick one by capability alone. Pinned by the Step 9 benchmark
# (see task-3-report.md); update this if a faster backend lands.
_MXFP8_DENSE_BACKEND = "cublaslt"


def grouped_mm(a, b, offs, bias=None, backend=None):
    # Triton's grouped kernel is bf16-only; every other dtype takes the reference.
    if backend is None and a.dtype is not torch.bfloat16:
        backend = "eager"
    return dispatch("gemm.grouped_mm", (a, b, offs, bias), {}, backend, device=a.device)


def int8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias=None, backend=None):
    return dispatch(
        "gemm.int8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {},
        backend,
        device=aq.device,
    )


def fp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias=None, backend=None):
    return dispatch(
        "gemm.fp8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {},
        backend,
        device=aq.device,
    )


def mxfp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias=None, backend=None):
    return dispatch(
        "gemm.mxfp8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {},
        backend or _MXFP8_DENSE_BACKEND,
        device=aq.device,
    )


def int8_scaled_grouped_mm(
    aq, bq, sa, sb, offs, out_dtype, block_size, bias=None, backend=None
):
    return dispatch(
        "gemm.int8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {},
        backend,
        device=aq.device,
    )


def fp8_scaled_grouped_mm(
    aq, bq, sa, sb, offs, out_dtype, block_size, bias=None, backend=None
):
    return dispatch(
        "gemm.fp8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {},
        backend,
        device=aq.device,
    )


def mxfp8_scaled_grouped_mm(
    aq, bq, sa, sb, offs, out_dtype, block_size, bias=None, backend=None
):
    return dispatch(
        "gemm.mxfp8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {},
        backend,
        device=aq.device,
    )
