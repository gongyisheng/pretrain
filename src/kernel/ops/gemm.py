import torch

import src.kernel.backends.cublaslt as _cublaslt_backend
import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.registry import KERNEL_REGISTRY
from src.kernel.selector import _platform_for, compile_safe_cache, dispatch

_cublaslt = getattr(_cublaslt_backend, "_gemm", None)


def _check_same_dtype(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dtype != b.dtype:
        raise ValueError(
            f"a and b must have the same dtype, got {a.dtype} and {b.dtype}"
        )


def _check_offs(offs: torch.Tensor) -> None:
    if not offs.is_contiguous():
        raise ValueError("offs must be contiguous")


def _check_contraction(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"contraction mismatch: a {a.shape[-1]}, b {b.shape[-2]}")


def _check_global_scale(
    gsa: torch.Tensor | None,
    gsb: torch.Tensor | None,
    n_groups: int,
) -> None:
    """Validate paired fp32 global scales for each group."""
    if (gsa is None) != (gsb is None):
        raise ValueError("global scale gsa and gsb must be given together")
    for name, g in (("gsa", gsa), ("gsb", gsb)):
        if g is None:
            continue
        if g.dtype is not torch.float32:
            raise ValueError(f"global scale {name} must be float32, got {g.dtype}")
        if g.shape != (n_groups,):
            raise ValueError(
                f"global scale {name} shape {tuple(g.shape)} != {(n_groups,)}"
            )


def _check_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    gsa: torch.Tensor | None,
    gsb: torch.Tensor | None,
    block_size: int,
    packed_e2m1: bool = False,
    require_nonzero_block_size: bool = False,
) -> None:
    if require_nonzero_block_size and block_size == 0:
        raise ValueError("block_size must be nonzero, got 0")
    logical_k = aq.shape[-1] * (2 if packed_e2m1 else 1)
    if packed_e2m1:
        if aq.dtype is not torch.uint8 or bq.dtype is not torch.uint8:
            raise ValueError("fp4_e2m1 operands must be packed in uint8 tensors")
        if sa.dtype is torch.float8_e8m0fnu or sb.dtype is torch.float8_e8m0fnu:
            raise ValueError("nvfp4 does not support float8_e8m0fnu block scales")
        if block_size < 0 or block_size % 16:
            raise ValueError(
                f"fp4_e2m1 block_size must be a positive multiple of 16, got {block_size}"
            )
        if logical_k % 16:
            raise ValueError(
                f"fp4_e2m1 logical contraction size must be a multiple of 16, got {logical_k}"
            )
    _check_global_scale(gsa, gsb, 1)
    _check_contraction(aq, bq)
    blocks = -(-logical_k // block_size) if block_size else 1
    if sa.shape[-1] != blocks:
        raise ValueError(f"sa block count {sa.shape[-1]} != {blocks}")
    if sb.shape[-2] != blocks:
        raise ValueError(f"sb block count {sb.shape[-2]} != {blocks}")


def _check_scaled_grouped_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    gsa: torch.Tensor | None,
    gsb: torch.Tensor | None,
    offs: torch.Tensor,
    block_size: int,
    packed_e2m1: bool = False,
    require_nonzero_block_size: bool = False,
) -> None:
    """Validate grouped scale shapes."""
    if require_nonzero_block_size and block_size == 0:
        raise ValueError("block_size must be nonzero, got 0")
    logical_k = aq.shape[-1] * (2 if packed_e2m1 else 1)
    if packed_e2m1:
        if aq.dtype is not torch.uint8 or bq.dtype is not torch.uint8:
            raise ValueError("fp4_e2m1 operands must be packed in uint8 tensors")
        if sa.dtype is torch.float8_e8m0fnu or sb.dtype is torch.float8_e8m0fnu:
            raise ValueError("nvfp4 does not support float8_e8m0fnu block scales")
        if block_size < 0 or block_size % 16:
            raise ValueError(
                f"fp4_e2m1 block_size must be a positive multiple of 16, got {block_size}"
            )
        if logical_k % 16:
            raise ValueError(
                f"fp4_e2m1 logical contraction size must be a multiple of 16, got {logical_k}"
            )
        if aq.ndim == 2 and bq.ndim == 2:
            valid = torch.all(offs.remainder(2) == 0) & (offs[-1] == logical_k)
            message = "nvfp4 ragged-K offsets must be even and end at logical K"
            if torch.compiler.is_compiling():
                torch._assert_async(valid, message)
            else:
                torch._assert(valid, message)
    _check_global_scale(gsa, gsb, offs.numel())
    _check_contraction(aq, bq)
    if aq.ndim == 2 and bq.ndim == 3:  # Ragged M.
        blocks = -(-logical_k // block_size) if block_size else 1
        expected_sa, expected_sb = (
            (aq.shape[0], blocks),
            (bq.shape[0], blocks, bq.shape[2]),
        )
    elif aq.ndim == 3 and bq.ndim == 2:  # Ragged N.
        blocks = -(-logical_k // block_size) if block_size else 1
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
    "nvfp4_scaled_mm",
    "int8_scaled_grouped_mm",
    "fp8_scaled_grouped_mm",
    "mxfp8_scaled_grouped_mm",
    "nvfp4_scaled_grouped_mm",
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
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None,
) -> str | None:
    """Choose an optimized backend, or defer to capability-based dispatch."""
    op = "gemm.mxfp8_scaled_mm"
    if (
        _cublaslt is not None
        and _cublaslt.supports_mxfp8_scaled_mm(
            aq, bq, sa, sb, out_dtype, block_size, bias
        )
        and _is_kernel_available(op, "cublaslt", aq.device)
    ):
        return "cublaslt"
    if _is_kernel_available(op, "triton", aq.device):
        return "triton"
    return None


def _select_nvfp4_scaled_mm_backend(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None,
) -> str | None:
    op = "gemm.nvfp4_scaled_mm"
    if (
        _cublaslt is not None
        and _cublaslt.supports_nvfp4_scaled_mm(
            aq, bq, sa, sb, out_dtype, block_size, bias
        )
        and _is_kernel_available(op, "cublaslt", aq.device)
    ):
        return "cublaslt"
    if _is_kernel_available(op, "triton", aq.device):
        return "triton"
    return None


def grouped_mm(a, b, offs, bias=None, backend=None):
    _check_offs(offs)
    _check_contraction(a, b)
    _check_same_dtype(a, b)
    if backend is None and a.dtype not in (torch.bfloat16, torch.float16):
        backend = "eager"
    return dispatch("gemm.grouped_mm", (a, b, offs, bias), {}, backend, device=a.device)


def int8_scaled_mm(
    aq, bq, sa, sb, out_dtype, block_size, bias=None, gsa=None, gsb=None, backend=None
):
    _check_scaled_mm(aq, bq, sa, sb, gsa, gsb, block_size)
    return dispatch(
        "gemm.int8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {"gsa": gsa, "gsb": gsb},
        backend,
        device=aq.device,
    )


def fp8_scaled_mm(
    aq, bq, sa, sb, out_dtype, block_size, bias=None, gsa=None, gsb=None, backend=None
):
    _check_scaled_mm(aq, bq, sa, sb, gsa, gsb, block_size)
    return dispatch(
        "gemm.fp8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {"gsa": gsa, "gsb": gsb},
        backend,
        device=aq.device,
    )


def mxfp8_scaled_mm(
    aq, bq, sa, sb, out_dtype, block_size, bias=None, gsa=None, gsb=None, backend=None
):
    _check_scaled_mm(
        aq, bq, sa, sb, gsa, gsb, block_size, require_nonzero_block_size=True
    )
    selected_backend = backend
    if selected_backend is None:
        selected_backend = _select_mxfp8_scaled_mm_backend(
            aq, bq, sa, sb, out_dtype, block_size, bias
        )
    return dispatch(
        "gemm.mxfp8_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {"gsa": gsa, "gsb": gsb},
        selected_backend,
        device=aq.device,
    )


def nvfp4_scaled_mm(
    aq, bq, sa, sb, out_dtype, block_size, bias=None, gsa=None, gsb=None, backend=None
):
    """Multiply packed NVFP4 matrices with block and optional global scales."""
    _check_scaled_mm(
        aq,
        bq,
        sa,
        sb,
        gsa,
        gsb,
        block_size,
        packed_e2m1=True,
        require_nonzero_block_size=True,
    )
    selected_backend = backend
    if selected_backend is None:
        selected_backend = _select_nvfp4_scaled_mm_backend(
            aq, bq, sa, sb, out_dtype, block_size, bias
        )
    return dispatch(
        "gemm.nvfp4_scaled_mm",
        (aq, bq, sa, sb, out_dtype, block_size, bias),
        {"gsa": gsa, "gsb": gsb},
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
    gsa=None,
    gsb=None,
    backend=None,
):
    _check_offs(offs)
    _check_scaled_grouped_mm(aq, bq, sa, sb, gsa, gsb, offs, block_size)
    return dispatch(
        "gemm.int8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {"gsa": gsa, "gsb": gsb},
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
    gsa=None,
    gsb=None,
    backend=None,
):
    _check_offs(offs)
    _check_scaled_grouped_mm(aq, bq, sa, sb, gsa, gsb, offs, block_size)
    return dispatch(
        "gemm.fp8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {"gsa": gsa, "gsb": gsb},
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
    gsa=None,
    gsb=None,
    backend=None,
):
    _check_offs(offs)
    _check_scaled_grouped_mm(
        aq,
        bq,
        sa,
        sb,
        gsa,
        gsb,
        offs,
        block_size,
        require_nonzero_block_size=True,
    )
    return dispatch(
        "gemm.mxfp8_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {"gsa": gsa, "gsb": gsb},
        backend,
        device=aq.device,
    )


def nvfp4_scaled_grouped_mm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    bias=None,
    gsa=None,
    gsb=None,
    backend=None,
):
    """Multiply grouped packed NVFP4 matrices with per-group scales."""
    _check_offs(offs)
    _check_scaled_grouped_mm(
        aq,
        bq,
        sa,
        sb,
        gsa,
        gsb,
        offs,
        block_size,
        packed_e2m1=True,
        require_nonzero_block_size=True,
    )
    return dispatch(
        "gemm.nvfp4_scaled_grouped_mm",
        (aq, bq, sa, sb, offs, out_dtype, block_size, bias),
        {"gsa": gsa, "gsb": gsb},
        backend,
        device=aq.device,
    )


SCALED_MM_OPS = {
    "gemm.int8_scaled_mm": int8_scaled_mm,
    "gemm.fp8_scaled_mm": fp8_scaled_mm,
    "gemm.mxfp8_scaled_mm": mxfp8_scaled_mm,
    "gemm.nvfp4_scaled_mm": nvfp4_scaled_mm,
    "gemm.int8_scaled_grouped_mm": int8_scaled_grouped_mm,
    "gemm.fp8_scaled_grouped_mm": fp8_scaled_grouped_mm,
    "gemm.mxfp8_scaled_grouped_mm": mxfp8_scaled_grouped_mm,
    "gemm.nvfp4_scaled_grouped_mm": nvfp4_scaled_grouped_mm,
}
