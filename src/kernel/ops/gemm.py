import torch

import src.kernel.backends.cublaslt  # noqa: F401
import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.registry import KERNEL_REGISTRY
from src.kernel.selector import _platform_for, compile_safe_cache, dispatch

try:  # Absent when the extension is unbuilt.
    from src.kernel.backends.cublaslt import gemm as _cublaslt
except ImportError:  # pragma: no cover - exercised only on non-CUDA builds
    _cublaslt = None


def _check_offs(offs: torch.Tensor) -> None:
    if not offs.is_contiguous():
        raise ValueError("offs must be contiguous")


def _check_contraction(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"contraction mismatch: a {a.shape[-1]}, b {b.shape[-2]}")


def _check_global_scale(
    ga: torch.Tensor | None, gb: torch.Tensor | None, n_groups: int
) -> None:
    """A global scale holds one fp32 value per group; a dense call has one group.

    Both operands carry one or neither: the Triton kernels gate the epilogue on a
    single `HAS_GLOBAL` constexpr, so a lone `gb` would be dropped there while the
    eager reference still applied it.
    """
    if (ga is None) != (gb is None):
        raise ValueError("global scale ga and gb must be given together")
    for name, g in (("ga", ga), ("gb", gb)):
        if g is None:
            continue
        if g.dtype is not torch.float32:
            raise ValueError(f"global scale {name} must be float32, got {g.dtype}")
        if g.shape != (n_groups,):
            raise ValueError(
                f"global scale {name} shape {tuple(g.shape)} != {(n_groups,)}"
            )


def _reject_global_scale(
    ga: torch.Tensor | None, gb: torch.Tensor | None, feature: str
) -> None:
    """E8M0 block scales absorb any global factor, so mxfp8 never carries one."""
    if ga is not None or gb is not None:
        raise ValueError(f"{feature} does not accept a global scale")


def _check_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    ga: torch.Tensor | None,
    gb: torch.Tensor | None,
    block_size: int,
) -> None:
    _check_global_scale(ga, gb, 1)
    _check_contraction(aq, bq)
    blocks = -(-aq.shape[-1] // block_size) if block_size else 1
    if sa.shape[-1] != blocks:
        raise ValueError(f"sa block count {sa.shape[-1]} != {blocks}")
    if sb.shape[-2] != blocks:
        raise ValueError(f"sb block count {sb.shape[-2]} != {blocks}")


def _check_scaled_grouped_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    ga: torch.Tensor | None,
    gb: torch.Tensor | None,
    offs: torch.Tensor,
    block_size: int,
) -> None:
    """Validate grouped scale shapes."""
    _check_global_scale(ga, gb, offs.numel())
    _check_contraction(aq, bq)
    if aq.ndim == 2 and bq.ndim == 3:  # Ragged M.
        blocks = -(-aq.shape[-1] // block_size) if block_size else 1
        expected_sa, expected_sb = (
            (aq.shape[0], blocks),
            (bq.shape[0], blocks, bq.shape[2]),
        )
    elif aq.ndim == 3 and bq.ndim == 2:  # Ragged N.
        blocks = -(-aq.shape[-1] // block_size) if block_size else 1
        expected_sa, expected_sb = (
            (aq.shape[0], aq.shape[1], blocks),
            (blocks, bq.shape[1]),
        )
    else:  # Ragged K requires group-specific block counts, so check consistency.
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


@compile_safe_cache
def _is_kernel_available(op: str, backend: str, device: torch.device) -> bool:
    """Return whether a registered backend supports the device."""
    platform = _platform_for(device.type, device.index)
    return any(
        spec.backend == backend and spec.is_available(platform)
        for spec in KERNEL_REGISTRY.implementations(op)
    )


def _select_mxfp8_scaled_mm_backend(
    aq: torch.Tensor,
    bq: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> str | None:
    """Choose an optimized backend, or defer to capability-based dispatch."""
    op = "gemm.mxfp8_scaled_mm"
    if (
        _cublaslt is not None
        and _cublaslt.supports_mxfp8_scaled_mm(aq, bq, out_dtype, block_size)
        and _is_kernel_available(op, "cublaslt", aq.device)
    ):
        return "cublaslt"
    if _is_kernel_available(op, "triton", aq.device):
        return "triton"
    return None


def grouped_mm(a, b, offs, bias=None, backend=None):
    _check_offs(offs)
    _check_contraction(a, b)
    if backend is None and a.dtype not in (torch.bfloat16, torch.float16):
        backend = "eager"
    return dispatch("gemm.grouped_mm", (a, b, offs, bias), {}, backend, device=a.device)


def int8_scaled_mm(
    aq, bq, sa, sb, out_dtype, block_size, bias=None, ga=None, gb=None, backend=None
):
    _check_scaled_mm(aq, bq, sa, sb, ga, gb, block_size)
    return dispatch(
        "gemm.int8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {"ga": ga, "gb": gb},
        backend,
        device=aq.device,
    )


def fp8_scaled_mm(
    aq, bq, sa, sb, out_dtype, block_size, bias=None, ga=None, gb=None, backend=None
):
    _check_scaled_mm(aq, bq, sa, sb, ga, gb, block_size)
    return dispatch(
        "gemm.fp8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {"ga": ga, "gb": gb},
        backend,
        device=aq.device,
    )


def mxfp8_scaled_mm(
    aq, bq, sa, sb, out_dtype, block_size, bias=None, ga=None, gb=None, backend=None
):
    # Rejected rather than applied: an e8m0 block scale absorbs a global factor, so
    # QuantizationConfig never resolves the pair and no kernel here implements it.
    _reject_global_scale(ga, gb, "mxfp8 GEMM")
    _check_scaled_mm(aq, bq, sa, sb, ga, gb, block_size)
    selected_backend = backend
    if selected_backend is None:
        selected_backend = _select_mxfp8_scaled_mm_backend(
            aq, bq, out_dtype, block_size
        )
    return dispatch(
        "gemm.mxfp8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {},
        selected_backend,
        device=aq.device,
    )


def int8_scaled_grouped_mm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    bias=None,
    ga=None,
    gb=None,
    backend=None,
):
    _check_offs(offs)
    _check_scaled_grouped_mm(aq, bq, sa, sb, ga, gb, offs, block_size)
    return dispatch(
        "gemm.int8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {"ga": ga, "gb": gb},
        backend,
        device=aq.device,
    )


def fp8_scaled_grouped_mm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    bias=None,
    ga=None,
    gb=None,
    backend=None,
):
    _check_offs(offs)
    _check_scaled_grouped_mm(aq, bq, sa, sb, ga, gb, offs, block_size)
    return dispatch(
        "gemm.fp8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {"ga": ga, "gb": gb},
        backend,
        device=aq.device,
    )


def mxfp8_scaled_grouped_mm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    bias=None,
    ga=None,
    gb=None,
    backend=None,
):
    _check_offs(offs)
    _reject_global_scale(ga, gb, "mxfp8 grouped GEMM")
    _check_scaled_grouped_mm(aq, bq, sa, sb, ga, gb, offs, block_size)
    return dispatch(
        "gemm.mxfp8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {},
        backend,
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
