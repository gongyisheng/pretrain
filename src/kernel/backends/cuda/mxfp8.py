from typing import Literal

import torch

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda


@register_kernel(
    op="quantize.quantize_mxfp8",
    backend="cuda",
    build="aot",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 9))}),
)
def quantize_mxfp8(
    x: torch.Tensor,
    contract_dim: int = -1,
    block_shape: tuple[int, int] = (1, 32),
    fmt: str = "fp8_e4m3",
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
    enable_global_scale: bool = False,
    stochastic_rounding: bool = False,
    output_layout: str = "row_major",
    scale_layout: str = "row_major",
    return_quantization_stats: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | Literal[False]
]:
    if fmt != "fp8_e4m3":
        raise ValueError("MXFP8 requires fp8_e4m3")
    if scale_dtype is not torch.float8_e8m0fnu or enable_global_scale is not False:
        raise ValueError("MXFP8 requires E8M0 scales without global scaling")
    if block_shape[1] not in (16, 32, 64, 128):
        raise ValueError("CUDA MXFP8 block extent must be 16, 32, 64, or 128")
    quantize = (
        torch.ops.aot_kernel.quantize_mxfp8_with_stats
        if return_quantization_stats
        else torch.ops.aot_kernel.quantize_mxfp8
    )
    result = quantize(
        x,
        contract_dim,
        block_shape[0],
        block_shape[1],
        stochastic_rounding,
        output_layout,
        scale_layout,
    )
    if return_quantization_stats:
        codes, scale, statistics = result
        return codes, scale, None, statistics.detach()
    codes, scale = result
    return codes, scale, None, False
