from functools import lru_cache

import torch

import src.kernel.backends.cublaslt  # noqa: F401
import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.registry import KERNEL_REGISTRY
from src.kernel.selector import _platform_for, dispatch

try:  # the registration gate leaves this absent when the extension is unbuilt
    from src.kernel.backends.cublaslt import gemm as _cublaslt
except ImportError:  # pragma: no cover - exercised only on non-CUDA builds
    _cublaslt = None


def _check_offs(offs: torch.Tensor) -> None:
    if not offs.is_contiguous():
        raise ValueError("offs must be contiguous")


def _check_contraction(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"contraction mismatch: a {a.shape[-1]}, b {b.shape[-2]}")


def _check_scaled_dense(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    block_size: int,
) -> None:
    _check_contraction(aq, bq)
    blocks = -(-aq.shape[-1] // block_size) if block_size else 1
    if sa.shape[-1] != blocks:
        raise ValueError(f"sa block count {sa.shape[-1]} != {blocks}")
    if sb.shape[-2] != blocks:
        raise ValueError(f"sb block count {sb.shape[-2]} != {blocks}")


def _check_scaled_grouped(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    block_size: int,
) -> None:
    """Mirrors `_grouped_scales_are_valid` (triton/gemm.py) -- the scale-block axis
    depends on which operand rank carries the ragged dim, not just on block_size.
    """
    _check_contraction(aq, bq)
    if aq.ndim == 2 and bq.ndim == 3:  # ragged-M: A's row axis, dense contraction
        blocks = -(-aq.shape[-1] // block_size) if block_size else 1
        expected_sa, expected_sb = (
            (aq.shape[0], blocks),
            (bq.shape[0], blocks, bq.shape[2]),
        )
    elif aq.ndim == 3 and bq.ndim == 2:  # ragged-N: B's column axis
        blocks = -(-aq.shape[-1] // block_size) if block_size else 1
        expected_sa, expected_sb = (
            (aq.shape[0], aq.shape[1], blocks),
            (blocks, bq.shape[1]),
        )
    else:  # ragged-K: the contraction itself, e.g. the wgrad -- block count is
        # per-group (Σ_g cdiv(k_g, block_size)) and unknowable without a host sync
        # on offs, so sa/sb are only checked for mutual consistency.
        if sa.shape[0] != aq.shape[0]:
            raise ValueError(f"sa rows {sa.shape[0]} != aq rows {aq.shape[0]}")
        if sb.shape[-1] != bq.shape[-1]:
            raise ValueError(f"sb cols {sb.shape[-1]} != bq cols {bq.shape[-1]}")
        if sa.shape[-1] != sb.shape[0]:
            raise ValueError(
                f"sa block count {sa.shape[-1]} != sb block count {sb.shape[0]}"
            )
        if block_size == 0 and sa.shape[-1] != offs.numel():
            raise ValueError(
                f"sa block count {sa.shape[-1]} != offs groups {offs.numel()}"
            )
        return
    if sa.shape != expected_sa:
        raise ValueError(
            f"sa block count {sa.shape[-1]} shape {tuple(sa.shape)} != {expected_sa}"
        )
    if sb.shape != expected_sb:
        raise ValueError(
            f"sb block count {sb.shape[-2]} shape {tuple(sb.shape)} != {expected_sb}"
        )


__all__ = [
    "grouped_mm",
    "int8_scaled_mm",
    "fp8_scaled_mm",
    "mxfp8_scaled_mm",
    "int8_scaled_grouped_mm",
    "fp8_scaled_grouped_mm",
    "mxfp8_scaled_grouped_mm",
    "SCALED_MM_OPS",
]


@lru_cache(maxsize=None)
def _cublaslt_eligible(op: str, device_type: str, device_index: int | None) -> bool:
    """Is cuBLASLt both built and hardware-eligible for `op` on this device?

    Registration covers the build axis -- an unbuilt `_C` registers no spec at all --
    and `is_available` covers the SM window. Cached per (op, device); the capability
    gate itself never runs on the hot path.
    """
    platform = _platform_for(device_type, device_index)
    return any(
        spec.backend == "cublaslt" and spec.is_available(platform)
        for spec in KERNEL_REGISTRY.implementations(op)
    )


def _mxfp8_backend(op: str, serves: bool, device: torch.device) -> str:
    """cuBLASLt for every mxfp8 call it can serve; Triton for the rest.

    cuBLASLt is the vendor kernel and wins decisively once a GEMM is compute-bound
    rather than launch-bound -- measured on sm120 at block_size=32: 18.7% faster at
    M,K,N=4096,4096,4096 and 31.5% at 8192,2048,2048, against 3-8% slower at the
    smaller shapes where both are launch-bound.

    Two axes decide it, and they are separate on purpose. `_cublaslt_eligible` is
    metadata -- the SM window plus whether the extension was built -- and is cached
    per device. `serves` is per-call and cannot be metadata: cuBLASLt reads bare
    32-wide e8m0 blocks, always returns bf16, and needs 16-byte-aligned operand
    strides, so a `block_size` of 64, an fp16 output, or an unaligned token count
    (M=250 leaves a 250-byte row stride) all fall to Triton, which serves every case.
    """
    if serves and _cublaslt_eligible(op, device.type, device.index):
        return "cublaslt"
    return "triton"


def grouped_mm(a, b, offs, bias=None, backend=None):
    _check_offs(offs)
    _check_contraction(a, b)
    # Triton's grouped kernel is bf16-only; every other dtype takes the reference.
    if backend is None and a.dtype is not torch.bfloat16:
        backend = "eager"
    return dispatch("gemm.grouped_mm", (a, b, offs, bias), {}, backend, device=a.device)


def int8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias=None, backend=None):
    _check_scaled_dense(aq, bq, sa, sb, block_size)
    return dispatch(
        "gemm.int8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {},
        backend,
        device=aq.device,
    )


def fp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias=None, backend=None):
    _check_scaled_dense(aq, bq, sa, sb, block_size)
    return dispatch(
        "gemm.fp8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {},
        backend,
        device=aq.device,
    )


def mxfp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias=None, backend=None):
    _check_scaled_dense(aq, bq, sa, sb, block_size)
    return dispatch(
        "gemm.mxfp8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {},
        backend
        or _mxfp8_backend(
            "gemm.mxfp8_scaled_mm",
            _cublaslt is not None
            and _cublaslt.serves_dense(aq, bq, out_dtype, block_size),
            aq.device,
        ),
        device=aq.device,
    )


def int8_scaled_grouped_mm(
    aq, bq, sa, sb, offs, out_dtype, block_size, bias=None, backend=None
):
    _check_offs(offs)
    _check_scaled_grouped(aq, bq, sa, sb, offs, block_size)
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
    _check_offs(offs)
    _check_scaled_grouped(aq, bq, sa, sb, offs, block_size)
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
    _check_offs(offs)
    _check_scaled_grouped(aq, bq, sa, sb, offs, block_size)
    return dispatch(
        "gemm.mxfp8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {},
        backend
        or _mxfp8_backend(
            "gemm.mxfp8_scaled_grouped_mm",
            _cublaslt is not None
            and _cublaslt.serves_grouped(aq, bq, out_dtype, block_size, bias),
            aq.device,
        ),
        device=aq.device,
    )


SCALED_MM_OPS = {
    "gemm.int8_scaled_mm": int8_scaled_mm,
    "gemm.fp8_scaled_mm": fp8_scaled_mm,
    "gemm.mxfp8_scaled_mm": mxfp8_scaled_mm,
    "gemm.int8_scaled_grouped_mm": int8_scaled_grouped_mm,
    "gemm.fp8_scaled_grouped_mm": fp8_scaled_grouped_mm,
    "gemm.mxfp8_scaled_grouped_mm": mxfp8_scaled_grouped_mm,
}
