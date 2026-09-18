"""Native NVFP4 quantization."""

from typing import Literal

import torch

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda


@register_kernel(
    op="quantize.quantize_nvfp4",
    backend="cuda",
    build="aot",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0), max_arch=(12, 99))}),
)
def quantize_nvfp4(
    x: torch.Tensor,
    contract_dim: int,
    block_shape: tuple[int, int],
    fmt: str = "fp4_e2m1",
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_global_scale: bool = True,
    stochastic_rounding: bool = False,
    output_layout: str = "row_major",
    scale_layout: str = "row_major",
    return_quantization_stats: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | Literal[False]
]:
    """Return CUDA quantization outputs and optional statistics.

    Statistics energy fields use atomic accumulation and are not bitwise deterministic.
    """
    if fmt not in ("fp4_e2m1", "fp4_e2m1_4over6"):
        raise ValueError("NVFP4 requires fp4_e2m1 or fp4_e2m1_4over6")
    qmax = 6.0 if fmt == "fp4_e2m1" else 4.0
    if scale_dtype is not torch.float8_e4m3fn:
        raise ValueError("CUDA NVFP4 requires E4M3 scales")
    if tuple(block_shape) not in ((1, 16), (16, 16)):
        raise ValueError("CUDA NVFP4 requires block_shape (1, 16) or (16, 16)")
    quantize = (
        torch.ops.aot_kernel.quantize_nvfp4_with_stats
        if return_quantization_stats
        else torch.ops.aot_kernel.quantize_nvfp4
    )
    result = quantize(
        x,
        contract_dim,
        block_shape,
        enable_global_scale,
        stochastic_rounding,
        qmax,
        output_layout,
        scale_layout,
    )
    if return_quantization_stats:
        codes, scale, global_scale, statistics = result
        return codes, scale, global_scale, statistics.detach()
    codes, scale, global_scale = result
    return codes, scale, global_scale, False
