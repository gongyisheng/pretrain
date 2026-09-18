"""Eager NVFP4 quantization kernels."""

from typing import Literal

import torch

from src.kernel.backends.eager.quantize import (
    EPS,
    _quantize_dense,
    _quantize_grouped,
    global_amax,
    global_amax_grouped,
    global_divisor,
    layout_codes,
    global_divisor_grouped,
    pack_e2m1,
)
from src.kernel.registry import register_kernel
from src.kernel.utils import to_swizzle_32_4_4


@register_kernel(
    op="quantize.quantize_nvfp4",
    backend="eager",
    build="eager",
    autograd=False,
    reference=True,
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
    if fmt not in ("fp4_e2m1", "fp4_e2m1_4over6"):
        raise ValueError("NVFP4 requires fp4_e2m1 or fp4_e2m1_4over6")
    if scale_layout not in ("row_major", "swizzled_32_4_4"):
        raise ValueError("unsupported NVFP4 scale layout")
    if scale_layout == "swizzled_32_4_4" and (
        x.ndim != 2 or block_shape[1] != 16 or scale_dtype is not torch.float8_e4m3fn
    ):
        raise ValueError("swizzled NVFP4 scales require rank-2 block-16 E4M3 scales")
    qmax = 6.0 if fmt == "fp4_e2m1" else 4.0
    values = x.float()
    global_scale = None
    if enable_global_scale and scale_dtype is torch.float8_e4m3fn:
        global_scale = (
            global_amax(values) / (qmax * float(torch.finfo(scale_dtype).max))
        ).clamp_min(EPS)
        values = values / global_divisor(global_scale, values)
    codes, scale, quantization_stats = _quantize_dense(
        values,
        contract_dim,
        torch.uint8,
        qmax,
        block_shape,
        scale_dtype,
        stochastic_rounding,
        source=x if return_quantization_stats else None,
        global_scale=global_scale,
    )
    codes = layout_codes(pack_e2m1(codes, contract_dim), output_layout)
    if scale_layout == "swizzled_32_4_4":
        scale = to_swizzle_32_4_4(scale if contract_dim == -1 else scale.t())
    elif block_shape != (0, 0):
        scale = scale.contiguous()
    return (
        codes,
        scale,
        global_scale,
        quantization_stats if return_quantization_stats else False,
    )


@register_kernel(
    op="quantize.quantize_nvfp4_grouped",
    backend="eager",
    build="eager",
    autograd=False,
    reference=True,
)
def quantize_nvfp4_grouped(
    x: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    block_shape: tuple[int, int],
    enable_global_scale: bool = True,
    stochastic_rounding: bool = False,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    qmax: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return packed grouped E2M1 codes, scales, and global scales."""
    values = x.float()
    global_scale = None
    if enable_global_scale and scale_dtype is torch.float8_e4m3fn:
        global_scale = (
            global_amax_grouped(values, offs, ragged_dim)
            / (qmax * float(torch.finfo(scale_dtype).max))
        ).clamp_min(EPS)
        values = values / global_divisor_grouped(global_scale, values, offs, ragged_dim)
    codes, scale = _quantize_grouped(
        values,
        offs,
        ragged_dim,
        contract_dim,
        torch.uint8,
        qmax,
        block_shape,
        scale_dtype,
        stochastic_rounding,
    )
    packed = pack_e2m1(codes, contract_dim)
    return packed.contiguous(), scale.contiguous(), global_scale
