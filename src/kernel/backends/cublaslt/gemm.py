import torch

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda
from src.kernel.utils import to_column_major, to_swizzle_32_4_4


_MXFP8_BLOCK_SIZE = 32
_NVFP4_BLOCK_SIZE = 16


def _check_layout(
    tensor: torch.Tensor,
    allowed_strides: tuple[tuple[int, ...], ...],
    feature: str,
) -> None:
    """Reject operand views the cuBLASLt kernel would silently miscompute on."""
    stride = tensor.stride()
    if tensor.numel() != 0 and stride not in allowed_strides:
        allowed = ", ".join(str(s) for s in allowed_strides)
        raise ValueError(f"{feature} has stride {stride}; requires one of {allowed}")
    element_size = tensor.element_size()
    misaligned = tuple(
        s * element_size for s in stride if s != 1 and (s * element_size) % 16 != 0
    )
    if misaligned:
        raise ValueError(
            f"{feature} has non-unit stride byte(s) {misaligned}; requires 16-byte "
            "alignment"
        )


def _check_dimension_multiple_of_16(tensor: torch.Tensor, feature: str) -> None:
    size = tensor.shape[-1]
    if size % 16 != 0:
        raise ValueError(f"{feature} has size {size}; requires a multiple of 16")


def _check_block_size(block_size: int, expected_val: int, feature: str) -> None:
    """cuBLASLt's scale descriptors fix the block width, unlike Triton's wider
    scale replication, so another width is silently wrong rather than rejected."""
    if block_size != expected_val:
        raise ValueError(
            f"{feature} requires block_size={expected_val}, got {block_size}"
        )


def _check_out_dtype(
    out_dtype: torch.dtype,
    supported: tuple[torch.dtype, ...],
    feature: str,
) -> None:
    if out_dtype not in supported:
        raise ValueError(f"{feature} does not support out_dtype={out_dtype}")


def _check_e4m3_element(tensor: torch.Tensor, feature: str) -> None:
    """This backend's descriptor is e4m3-only."""
    if tensor.dtype is not torch.float8_e4m3fn:
        raise ValueError(f"{feature} requires float8_e4m3fn, got {tensor.dtype}")


def _check_e8m0_element(tensor: torch.Tensor, feature: str) -> None:
    if tensor.dtype is not torch.float8_e8m0fnu:
        raise ValueError(f"{feature} requires float8_e8m0fnu, got {tensor.dtype}")


def _check_packed_e2m1_element(tensor: torch.Tensor, feature: str) -> None:
    if tensor.dtype is not torch.uint8:
        raise ValueError(f"{feature} requires packed uint8 values, got {tensor.dtype}")


def _check_scale_shapes(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    block_size: int,
    feature: str,
) -> None:
    blocks = -(-aq.shape[1] // block_size)
    expected_sa = (aq.shape[0], blocks)
    expected_sb = (blocks, bq.shape[1])
    if sa.shape != expected_sa or sb.shape != expected_sb:
        raise ValueError(
            f"{feature} scales must have shapes {expected_sa} and {expected_sb}"
        )


def _check_operand_shapes(
    aq: torch.Tensor,
    bq: torch.Tensor,
    feature: str,
) -> None:
    if aq.ndim != 2 or bq.ndim != 2:
        raise ValueError(f"{feature} requires rank-2 operands")
    if aq.shape[1] != bq.shape[0]:
        raise ValueError(f"{feature} contraction dimensions must match")


def _check_bias(
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    n: int,
    feature: str,
) -> None:
    if bias is None:
        return
    if bias.dtype is not out_dtype:
        raise ValueError(f"{feature} bias dtype must match out_dtype")
    if bias.shape != (n,):
        raise ValueError(f"{feature} bias must have shape ({n},)")
    if not bias.is_contiguous():
        raise ValueError(f"{feature} bias must be contiguous")


def _check_mxfp8_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None,
) -> None:
    """Everything this backend's dense kernel requires beyond its SM window."""
    _check_operand_shapes(aq, bq, "MXFP8 GEMM")
    _check_out_dtype(out_dtype, (torch.bfloat16,), "MXFP8 GEMM")
    _check_block_size(block_size, _MXFP8_BLOCK_SIZE, "MXFP8 GEMM")
    _check_layout(aq, ((aq.shape[1], 1),), "MXFP8 GEMM A")
    _check_layout(bq, ((bq.shape[1], 1), (1, bq.shape[0])), "MXFP8 GEMM B")
    _check_dimension_multiple_of_16(aq, "MXFP8 GEMM A")
    _check_dimension_multiple_of_16(bq, "MXFP8 GEMM B")
    _check_e4m3_element(aq, "MXFP8 GEMM A")
    _check_e4m3_element(bq, "MXFP8 GEMM B")
    _check_e8m0_element(sa, "MXFP8 GEMM A scale")
    _check_e8m0_element(sb, "MXFP8 GEMM B scale")
    _check_scale_shapes(aq, bq, sa, sb, block_size, "MXFP8 GEMM")
    _check_bias(bias, out_dtype, bq.shape[1], "MXFP8 GEMM")


def supports_mxfp8_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None,
) -> bool:
    """Can this backend support the call? Routed through the kernel's own guards so
    the routing decision and the kernel's contract cannot drift apart.

    The SM window is a `capabilities` matter and is checked separately; what lives
    here is per-call and inexpressible as metadata -- scale width, output dtype, and
    operand alignment (an unaligned token count leaves a non-16-byte row stride).
    """
    try:
        _check_mxfp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)
    except ValueError:
        return False
    return True


@register_kernel(
    op="gemm.mxfp8_scaled_mm",
    backend="cublaslt",
    build="aot",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0))}),
)
def scaled_mm_mxfp8(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
    gsa: torch.Tensor | None = None,
    gsb: torch.Tensor | None = None,
) -> torch.Tensor:
    _check_mxfp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)
    del out_dtype, block_size
    return torch.ops.aot_kernel._scaled_mm_mxfp8_cublaslt(
        aq,
        to_column_major(bq),
        to_swizzle_32_4_4(sa),
        to_swizzle_32_4_4(sb.t()),
        bias,
        gsa,
        gsb,
    )


def _check_nvfp4_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None,
) -> None:
    _check_operand_shapes(aq, bq, "NVFP4 GEMM")
    _check_out_dtype(
        out_dtype,
        (torch.bfloat16, torch.float16, torch.float32),
        "NVFP4 GEMM",
    )
    _check_block_size(block_size, _NVFP4_BLOCK_SIZE, "NVFP4 GEMM")
    _check_layout(aq, ((aq.shape[1], 1),), "NVFP4 GEMM A")
    _check_layout(bq, ((bq.shape[1], 1), (1, aq.shape[1])), "NVFP4 GEMM B")
    _check_dimension_multiple_of_16(aq, "NVFP4 GEMM A")
    _check_dimension_multiple_of_16(bq, "NVFP4 GEMM B")
    _check_packed_e2m1_element(aq, "NVFP4 GEMM A")
    _check_packed_e2m1_element(bq, "NVFP4 GEMM B")
    _check_e4m3_element(sa, "NVFP4 GEMM A scale")
    _check_e4m3_element(sb, "NVFP4 GEMM B scale")
    _check_scale_shapes(aq, bq, sa, sb, block_size // 2, "NVFP4 GEMM")
    if bias is not None:
        if out_dtype is torch.float32:
            raise ValueError("NVFP4 GEMM does not support float32 output with bias")
    _check_bias(bias, out_dtype, bq.shape[1], "NVFP4 GEMM")


def supports_nvfp4_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None,
) -> bool:
    try:
        _check_nvfp4_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)
    except ValueError:
        return False
    return True


@register_kernel(
    op="gemm.nvfp4_scaled_mm",
    backend="cublaslt",
    build="aot",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0))}),
)
def nvfp4_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
    gsa: torch.Tensor | None = None,
    gsb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense NVFP4 GEMM through the project-owned cuBLASLt kernel."""
    _check_nvfp4_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)
    return torch.ops.aot_kernel._scaled_mm_nvfp4_cublaslt(
        aq,
        to_column_major(bq),
        to_swizzle_32_4_4(sa),
        to_swizzle_32_4_4(sb.t()),
        out_dtype,
        bias,
        gsa,
        gsb,
    )
