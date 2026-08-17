import torch

import src.kernel.backends.cublaslt  # noqa: F401
import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry
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


def _resolve_mxfp8_dense_backend(registry: KernelRegistry) -> str:
    """cuBLASLt won the Step 9 benchmark (see task-3-report.md) for
    gemm.mxfp8_scaled_mm, so it is the default pin below -- but only when the AOT
    extension is actually built and its spec registered. On a build without that
    extension the spec never lands in `registry`, and pinning "cublaslt" anyway
    would make every mxfp8_scaled_mm() call raise KernelSelectionError; fall back to
    Triton instead."""
    has_cublaslt = any(
        spec.backend == "cublaslt"
        for spec in registry.implementations("gemm.mxfp8_scaled_mm")
    )
    return "cublaslt" if has_cublaslt else "triton"


# Triton and cuBLASLt are both eligible for gemm.mxfp8_scaled_mm on sm100+, so
# dispatch can't pick one by capability alone; this is the default backend used
# by mxfp8_scaled_mm() below when a caller doesn't ask for one explicitly.
_MXFP8_DENSE_BACKEND = _resolve_mxfp8_dense_backend(KERNEL_REGISTRY)

# cuBLASLt's mxfp8 kernels hard-require block_size=32 and 16-byte-aligned strides,
# and the dense kernel silently ignores the requested out_dtype -- constraints the
# quant layer's callers (src/quant/linear.py, src/quant/moe.py) cannot guarantee:
# arbitrary block sizes, transposed/unaligned-token wgrad shapes, and ragged expert
# groups. Both of them force Triton for mxfp8 regardless of the pins above, using
# these two constants instead of an unlabeled string so the override is discoverable
# from here rather than buried at each call site.
_MXFP8_DENSE_BACKEND_FOR_QUANT = "triton"

# gemm.mxfp8_scaled_grouped_mm's cuBLASLt kernel is registered only for
# cuda(min_arch=(10,0), max_arch=(11,0)); Triton has no ceiling. On this repo's
# sm120 hardware the ceiling excludes cuBLASLt, so quantized_grouped_gemm forcing
# Triton is a no-op here -- but on SM100/SM110 both backends are eligible, and this
# pin forecloses cuBLASLt there too. The pre-Task-3 selector reached cuBLASLt on
# that hardware via a per-call shape predicate; a shape-conditional replacement is
# untested on hardware neither author of this pin can exercise, so it isn't
# attempted here. Task 7 owns deciding that routing with real SM100/SM110 evidence.
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
