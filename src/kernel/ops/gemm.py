import torch

from src.kernel.backends import cublaslt as _cublaslt_backend  # noqa: F401
from src.kernel.backends import eager as _eager_backend  # noqa: F401
from src.kernel.backends import torch as _torch_backend  # noqa: F401
from src.kernel.backends import triton as _triton_backend  # noqa: F401
from src.kernel.selector import KernelSelectionError, dispatch
from src.kernel.utils import check_torch_tensors

__all__ = ["grouped_gemm", "scaled_gemm", "scaled_grouped_gemm"]


_SCALE_DTYPES = {
    "fp32": torch.float32,
    "fp8_e8m0": torch.float8_e8m0fnu,
}


def _normalize_scale_dtype(scale_dtype):
    if scale_dtype is None:
        return torch.float32
    if isinstance(scale_dtype, torch.dtype):
        return scale_dtype
    if scale_dtype not in _SCALE_DTYPES:
        raise KernelSelectionError(f"unknown scale_dtype {scale_dtype!r}")
    return _SCALE_DTYPES[scale_dtype]


def _require_torch_tensors(tensors: tuple[object, ...], feature: str) -> None:
    result = check_torch_tensors(tensors, feature)
    if not result.ok:
        raise KernelSelectionError(result.reason)


def grouped_gemm(a, b, offs, bias=None, backend="auto"):
    tensors = (a, b, offs) if bias is None else (a, b, offs, bias)
    _require_torch_tensors(tensors, "grouped GEMM")
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
    tensors = (aq, bq, sa, sb) if bias is None else (aq, bq, sa, sb, bias)
    _require_torch_tensors(tensors, "scaled GEMM")
    scale_dtype = _normalize_scale_dtype(scale_dtype)
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
    tensors = (aq, bq, sa, sb, offs) if bias is None else (aq, bq, sa, sb, offs, bias)
    _require_torch_tensors(tensors, "scaled grouped GEMM")
    scale_dtype = _normalize_scale_dtype(scale_dtype)
    return dispatch(
        "gemm.scaled_grouped",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias, scale_dtype),
        {},
        backend,
    )
