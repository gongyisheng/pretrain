"""Eager MXFP8 quantization kernels."""

import torch

from src.kernel.backends.eager.quantize import (
    _quantize_dense,
    _quantize_grouped,
    layout_codes,
)
from src.kernel.registry import register_kernel


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
    stochastic_rounding: bool = False,
    output_layout: str = "row_major",
    return_quantization_stats: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
):
    """Return E4M3 codes, E8M0 block scales, and no global scale."""
    del fmt
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
    if return_quantization_stats:
        return codes, scale.contiguous(), None, quantization_stats
    return codes, scale.contiguous(), None


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
