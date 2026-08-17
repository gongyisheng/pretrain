import torch

import src.kernel.backends.cublaslt  # noqa: F401
import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.selector import dispatch

__all__ = [
    "grouped_gemm",
    "scaled_gemm",
    "scaled_grouped_gemm",
]


def grouped_gemm(a, b, offs, bias=None, backend=None):
    # Triton's grouped kernel is bf16-only; every other dtype takes the reference.
    if backend is None and a.dtype is not torch.bfloat16:
        backend = "eager"
    return dispatch("gemm.grouped", (a, b, offs, bias), {}, backend, device=a.device)


def scaled_gemm(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    scale_dtype,
    bias=None,
    backend=None,
):
    return dispatch(
        "gemm.scaled",
        (aq, bq, sa, sb, out_dtype, block_size, scale_dtype, bias),
        {},
        backend,
        device=aq.device,
    )


def scaled_grouped_gemm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    scale_dtype,
    bias=None,
    backend=None,
):
    return dispatch(
        "gemm.scaled_grouped",
        (aq, bq, sa, sb, offs, out_dtype, block_size, scale_dtype, bias),
        {},
        backend,
        device=aq.device,
    )
