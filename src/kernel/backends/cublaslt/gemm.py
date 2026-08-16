import torch

from src.kernel.ops.gemm import can_implement_scaled_gemm_cublaslt
from src.kernel.registry import register_kernel
from src.kernel.utils import to_column_major, to_swizzle_32_4_4


@register_kernel(
    op="gemm.scaled",
    backend="cublaslt",
    can_implement=can_implement_scaled_gemm_cublaslt,
    build="aot",
    autograd=False,
)
def scaled_gemm_mxfp8(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    scale_dtype,
    bias=None,
):
    del out_dtype, block_size, scale_dtype
    return torch.ops.aot_kernel._scaled_gemm_mxfp8_cublaslt(
        aq,
        to_column_major(bq),
        to_swizzle_32_4_4(sa),
        to_swizzle_32_4_4(sb.t()),
        bias,
    )
