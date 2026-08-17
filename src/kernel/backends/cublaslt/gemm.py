import torch

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda
from src.kernel.utils import to_column_major, to_swizzle_32_4_4


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


def _check_block_size(block_size: int, feature: str) -> None:
    """cuBLASLt reads the scale tensors as bare 32-wide e8m0 blocks; unlike Triton
    it does not replicate a wider scale across consecutive blocks, so any other
    width is silently wrong rather than rejected."""
    if block_size != 32:
        raise ValueError(f"{feature} requires block_size=32, got {block_size}")


def _check_out_dtype(out_dtype: torch.dtype, feature: str) -> None:
    """The kernel always returns bf16; unlike Triton it does not cast, so any other
    request is silently wrong rather than rejected."""
    if out_dtype is not torch.bfloat16:
        raise ValueError(f"{feature} requires out_dtype=bfloat16, got {out_dtype}")


@register_kernel(
    op="gemm.mxfp8_scaled_mm",
    backend="cublaslt",
    build="aot",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0))}),
)
def scaled_gemm_mxfp8(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _check_out_dtype(out_dtype, "MXFP8 GEMM")
    del out_dtype
    _check_block_size(block_size, "MXFP8 GEMM")
    del block_size
    _check_layout(aq, ((aq.shape[1], 1),), "MXFP8 GEMM A")
    _check_layout(bq, ((bq.shape[1], 1), (1, bq.shape[0])), "MXFP8 GEMM B")
    _check_dimension_multiple_of_16(aq, "MXFP8 GEMM A")
    _check_dimension_multiple_of_16(bq, "MXFP8 GEMM B")
    return torch.ops.aot_kernel._scaled_gemm_mxfp8_cublaslt(
        aq,
        to_column_major(bq),
        to_swizzle_32_4_4(sa),
        to_swizzle_32_4_4(sb.t()),
        bias,
    )


@register_kernel(
    op="gemm.mxfp8_scaled_grouped_mm",
    backend="cublaslt",
    build="aot",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0), max_arch=(11, 0))}),
)
def scaled_grouped_gemm_mxfp8(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _check_out_dtype(out_dtype, "MXFP8 grouped GEMM")
    del out_dtype
    _check_block_size(block_size, "MXFP8 grouped GEMM")
    del block_size
    if bias is not None:
        raise ValueError("cuBLASLt MXFP8 grouped GEMM does not support bias")
    _check_layout(aq, ((aq.shape[1], 1),), "MXFP8 grouped GEMM A")
    _check_layout(
        bq,
        (
            (bq.shape[1] * bq.shape[2], bq.shape[2], 1),
            (bq.shape[1] * bq.shape[2], 1, bq.shape[1]),
        ),
        "MXFP8 grouped GEMM B",
    )
    _check_dimension_multiple_of_16(aq, "MXFP8 grouped GEMM A")
    _check_dimension_multiple_of_16(bq, "MXFP8 grouped GEMM B")
    return torch.ops.aot_kernel._scaled_grouped_gemm_mxfp8_cublaslt(
        aq,
        to_column_major(bq),
        sa,
        sb,
        offs,
    )
