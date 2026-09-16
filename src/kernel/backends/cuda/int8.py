import torch

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda


@register_kernel(
    op="quantize.quantize_int8",
    backend="cuda",
    build="aot",
    autograd=False,
    capabilities=frozenset({cuda()}),
)
def quantize_int8(
    x: torch.Tensor,
    contract_dim: int,
    block_shape: tuple[int, int],
    bits: int = 8,
    stochastic_rounding: bool = False,
    scale_dtype: torch.dtype = torch.float32,
    enable_global_scale: bool = False,
    output_layout: str = "row_major",
    return_quantization_stats: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, None]
    | tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]
):
    """Quantize 4–8 effective bits with FP32 reductions and scales."""
    if scale_dtype is not torch.float32 or enable_global_scale:
        raise ValueError("CUDA INT8 supports only FP32 scales without global scaling")
    quantize = (
        torch.ops.aot_kernel.quantize_int8_with_stats
        if return_quantization_stats
        else torch.ops.aot_kernel.quantize_int8
    )
    result = quantize(
        x,
        contract_dim,
        block_shape[0],
        block_shape[1],
        bits,
        stochastic_rounding,
        output_layout,
    )
    if return_quantization_stats:
        codes, scale, statistics = result
        return codes, scale, None, statistics.detach()
    codes, scale = result
    return codes, scale, None
