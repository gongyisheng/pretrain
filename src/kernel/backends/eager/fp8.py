"""Eager FP8 quantization kernels."""

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
    if fmt not in ("fp8_e4m3", "fp8_e5m2"):
        raise ValueError("FP8 requires fp8_e4m3 or fp8_e5m2")
    dtype = torch.float8_e4m3fn if fmt == "fp8_e4m3" else torch.float8_e5m2
    if scale_layout != "row_major":
        raise ValueError("FP8 requires row_major scales")
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
    codes = layout_codes(codes, output_layout)
    if block_shape != (0, 0):
        scale = scale.contiguous()
    return (
        codes,
        scale,
        global_scale,
        quantization_stats if return_quantization_stats else False,
    )


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
