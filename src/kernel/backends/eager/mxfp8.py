"""Eager MXFP8 quantization kernels."""

from typing import Literal

import torch

from src.kernel.backends.eager.quantize import (
    _quantize_dense,
    _quantize_grouped,
    layout_codes,
)
from src.kernel.registry import register_kernel
from src.kernel.utils import to_swizzle_32_4_4


@register_kernel(
    op="quantize.quantize_mxfp8",
    backend="eager",
    build="eager",
    autograd=False,
    reference=True,
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
    if scale_layout not in ("row_major", "swizzled_32_4_4"):
        raise ValueError("unsupported MXFP8 scale layout")
    if scale_layout == "swizzled_32_4_4" and (x.ndim != 2 or block_shape[1] != 32):
        raise ValueError("swizzled MXFP8 scales require rank-2 block-32 quantization")
    codes, scale, quantization_stats = _quantize_dense(
        x.float(),
        contract_dim,
        torch.float8_e4m3fn,
        448.0,
        block_shape,
        torch.float8_e8m0fnu,
        stochastic_rounding=stochastic_rounding,
        source=x if return_quantization_stats else None,
        global_scale=None,
    )
    codes = layout_codes(codes, output_layout)
    scale = (
        to_swizzle_32_4_4(scale if contract_dim == -1 else scale.t())
        if scale_layout == "swizzled_32_4_4"
        else scale.contiguous()
    )
    return (
        codes,
        scale,
        None,
        quantization_stats if return_quantization_stats else False,
    )


@register_kernel(
    op="quantize.quantize_mxfp8_grouped",
    backend="eager",
    build="eager",
    autograd=False,
    reference=True,
)
def quantize_mxfp8_grouped(
    x: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    block_shape: tuple[int, int],
    stochastic_rounding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Return grouped E4M3 codes, E8M0 scales, and no global scale."""
    codes, scale = _quantize_grouped(
        x.float(),
        offs,
        ragged_dim,
        contract_dim,
        torch.float8_e4m3fn,
        448.0,
        block_shape,
        torch.float8_e8m0fnu,
        stochastic_rounding,
    )
    return codes.contiguous(), scale.contiguous(), None
