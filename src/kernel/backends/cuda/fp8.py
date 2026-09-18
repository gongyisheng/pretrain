from typing import Literal

import torch

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda


@register_kernel(
    op="quantize.quantize_fp8",
    backend="cuda",
    build="aot",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 9))}),
)
def quantize_fp8(
    x: torch.Tensor,
    contract_dim: int,
    block_shape: tuple[int, int],
    fmt: str = "fp8_e4m3",
    scale_dtype: torch.dtype = torch.float32,
    enable_global_scale: bool = False,
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
    if fmt not in ("fp8_e4m3", "fp8_e5m2"):
        raise ValueError("FP8 requires fp8_e4m3 or fp8_e5m2")
    dtype = torch.float8_e4m3fn if fmt == "fp8_e4m3" else torch.float8_e5m2
    if scale_layout != "row_major":
        raise ValueError("FP8 requires row_major scales")
    if scale_dtype is not torch.float32 or enable_global_scale:
        raise ValueError("CUDA FP8 supports only FP32 scales without global scaling")
    quantize = (
        torch.ops.aot_kernel.quantize_fp8_with_stats
        if return_quantization_stats
        else torch.ops.aot_kernel.quantize_fp8
    )
    result = quantize(
        x,
        contract_dim,
        dtype,
        block_shape[0],
        block_shape[1],
        stochastic_rounding,
        output_layout,
    )
    if return_quantization_stats:
        codes, scale, statistics = result
        return codes, scale, None, statistics.detach()
    codes, scale = result
    return codes, scale, None, False
