import torch

import src.kernel.backends.cublaslt  # noqa: F401
import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.selector import dispatch


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
        backend or _MXFP8_DENSE_BACKEND,
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
        backend or _MXFP8_GROUPED_BACKEND,
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
