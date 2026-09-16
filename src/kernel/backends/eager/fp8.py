"""Eager FP8 quantization kernels."""

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
)
from src.kernel.registry import register_kernel


@register_kernel(
    op="quantize.quantize_fp8",
    backend="eager",
    build="eager",
    autograd=False,
    reference=True,
)
def quantize_fp8(
    x: torch.Tensor,
    contract_dim: int,
    dtype: torch.dtype,
    block_shape: tuple[int, int],
    stochastic_rounding: bool = False,
    scale_dtype: torch.dtype = torch.float32,
    enable_global_scale: bool = False,
    preserve_strides: bool = False,
    output_layout: str = "row_major",
    return_quantization_stats: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
):
    """Return FP8 codes, dequantization scales, and an optional global scale."""
    values = x.float()
    qmax = float(torch.finfo(dtype).max)
    global_scale = None
    if enable_global_scale and scale_dtype is torch.float8_e4m3fn:
        global_scale = (
            global_amax(values) / (qmax * float(torch.finfo(scale_dtype).max))
        ).clamp_min(EPS)
        values = values / global_divisor(global_scale, values)
    codes, scale, quantization_stats = _quantize_dense(
        values,
        contract_dim,
        dtype,
        qmax,
        block_shape,
        scale_dtype,
        stochastic_rounding=stochastic_rounding,
        source=x if return_quantization_stats else None,
        global_scale=global_scale,
    )
    if preserve_strides and output_layout != "row_major":
        raise ValueError("preserve_strides cannot be combined with column_major output")
    if not preserve_strides:
        codes = layout_codes(codes, output_layout)
    if block_shape != (0, 0) and not preserve_strides:
        scale = scale.contiguous()
    if return_quantization_stats:
        return codes, scale, global_scale, quantization_stats
    return codes, scale, global_scale


@register_kernel(
    op="quantize.quantize_fp8_grouped",
    backend="eager",
    build="eager",
    autograd=False,
    reference=True,
)
def quantize_fp8_grouped(
    x: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    dtype: torch.dtype,
    block_shape: tuple[int, int],
    stochastic_rounding: bool = False,
    scale_dtype: torch.dtype = torch.float32,
    enable_global_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return grouped FP8 codes, scales, and an optional global scale."""
    values = x.float()
    qmax = float(torch.finfo(dtype).max)
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
        dtype,
        qmax,
        block_shape,
        scale_dtype,
        stochastic_rounding,
    )
    if block_shape[1]:
        codes = codes.contiguous()
    return codes, scale.contiguous(), global_scale
