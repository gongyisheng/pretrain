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

# cuBLASLt measured ~5% faster than Triton for gemm.mxfp8_scaled_mm on this hardware
# (RTX 5060 Ti sm120, M,K,N=4096,2048,2048, block_size=32 -- see the Step 9 benchmark
# in task-3-report.md), but it only accepts block_size == 32 with 16-byte-aligned
# strides and silently ignores out_dtype. Callers that legitimately pass
# block_size=64 or unaligned/transposed shapes (e.g. src/quant/linear.py's wgrad)
# would hit a ValueError or a wrong dtype, so cuBLASLt cannot be the *default* --
# only Triton serves every caller of this op. Pass backend="cublaslt" explicitly
# when the caller's shapes are known to satisfy those constraints; the capability
# gate permits it since an explicit backend bypasses eligibility filtering.
_MXFP8_DENSE_BACKEND = "triton"

# gemm.mxfp8_scaled_grouped_mm's cuBLASLt kernel is registered only for
# cuda(min_arch=(10,0), max_arch=(11,0)); Triton has no ceiling. On this repo's
# sm120 hardware the ceiling already excludes cuBLASLt, so this default is a no-op
# here -- but on SM100/SM110 both backends are eligible, and this pin forecloses
# cuBLASLt there too. The pre-Task-3 selector reached cuBLASLt on that hardware via
# a per-call shape predicate; a shape-conditional replacement is untested on
# hardware neither author of this pin can exercise, so it isn't attempted here.
# Task 7 owns deciding that routing with real SM100/SM110 evidence.
_MXFP8_GROUPED_BACKEND = "triton"


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
