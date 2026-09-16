"""Native NVFP4 quantization."""

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
    enable_global_scale: bool = True,
    stochastic_rounding: bool = False,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    qmax: float = 6.0,
    output_layout: str = "row_major",
    return_quantization_stats: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]
):
    """Reduce scales, round, and pack on the current CUDA stream."""
    if scale_dtype is not torch.float8_e4m3fn or qmax not in (4.0, 6.0):
        raise ValueError("CUDA NVFP4 requires E4M3 scales and qmax 4 or 6")
    if tuple(block_shape) not in ((1, 16), (16, 16)):
        raise ValueError("CUDA NVFP4 requires block_shape (1, 16) or (16, 16)")
    if return_quantization_stats:
        codes, scale, global_scale, stats = (
            torch.ops.aot_kernel.quantize_nvfp4_with_stats(
                x,
                contract_dim,
                block_shape,
                enable_global_scale,
                stochastic_rounding,
                qmax,
                output_layout,
            )
        )
        return codes, scale, global_scale, stats.detach()
    return torch.ops.aot_kernel.quantize_nvfp4(
        x,
        contract_dim,
        block_shape,
        enable_global_scale,
        stochastic_rounding,
        qmax,
        output_layout,
    )
