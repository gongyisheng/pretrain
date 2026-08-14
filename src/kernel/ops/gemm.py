from src.kernel.backends import eager as _eager_backend  # noqa: F401
from src.kernel.backends import torch as _torch_backend  # noqa: F401
from src.kernel.backends import triton as _triton_backend  # noqa: F401
from src.kernel.selector import dispatch

__all__ = ["grouped_gemm", "scaled_gemm", "scaled_grouped_gemm"]


def grouped_gemm(a, b, offs, bias=None, backend="auto"):
    return dispatch("gemm.grouped", (a, b, offs, bias), {}, backend)


def scaled_gemm(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    bias=None,
    scale_dtype=None,
    backend="auto",
):
    return dispatch(
        "gemm.scaled",
        (aq, bq, sa, sb, out_dtype, block_size, bias, scale_dtype),
        {},
        backend,
    )


def scaled_grouped_gemm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    bias=None,
    scale_dtype=None,
    backend="auto",
):
    return dispatch(
        "gemm.scaled_grouped",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias, scale_dtype),
        {},
        backend,
    )
