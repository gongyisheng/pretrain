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
    stochastic_rounding: bool = False,
    output_layout: str = "row_major",
    return_quantization_stats: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, None]
    | tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]
):
    if block_shape[1] not in (16, 32, 64, 128):
        raise ValueError("CUDA MXFP8 block extent must be 16, 32, 64, or 128")
    if return_quantization_stats:
        codes, scale, stats = torch.ops.aot_kernel.quantize_mxfp8_with_stats(
            x,
            contract_dim,
            block_shape[0],
            block_shape[1],
            stochastic_rounding,
            output_layout,
        )
        return codes, scale, None, stats.detach()
    codes, scale = torch.ops.aot_kernel.quantize_mxfp8(
        x,
        contract_dim,
        block_shape[0],
        block_shape[1],
        stochastic_rounding,
        output_layout,
    )
    return codes, scale, None
