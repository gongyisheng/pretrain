import copy

import torch

from src.kernel.ops.gemm import SCALED_MM_OPS, grouped_mm
from src.layers.mlp import SparseMoEBlock
from src.metrics.quant import QuantizationStats, record_operand
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.rotation import Rotation
from src.quant.utils import is_quantized, scaled_grouped_mm_op
from src.utils.config import QuantizationConfig


def quantized_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    a_fmt: str,
    b_fmt: str,
    out_dtype: torch.dtype,
    scale_cfg: dict,
    bias: torch.Tensor | None = None,
    a_stochastic_rounding: bool = False,
    b_stochastic_rounding: bool = False,
    a_stats: QuantizationStats | None = None,
    b_stats: QuantizationStats | None = None,
    rotation: Rotation | None = None,
) -> torch.Tensor:
    """Quantized grouped GEMM for ragged M, K, or N layouts.

    Operand ranks select the layout. `bias` has shape (E, N), broadcasts over output
    rows, and is not quantized.
    """
    block_size = scale_cfg["block_shape"][1]
    op = scaled_grouped_mm_op(
        a_fmt,
        b_fmt,
        scale_cfg["scale_dtype"],
        scale_cfg["block_shape"],
    )
    ragged_k = a.ndim == 2 and b.ndim == 2

    # Map the operand containing the ragged axis.
    if a.ndim == 3:
        # Ragged N: only B is mapped on its outer axis.
        src_a, contract_a, a_offs, a_ragged_dim = a, -1, None, None
        b_offs, b_ragged_dim = offs, -1
    elif ragged_k:
        # Ragged K: map both operands and transpose A for quantization.
        src_a, contract_a, a_offs, a_ragged_dim = a.mT, -2, offs, -2
        b_offs, b_ragged_dim = offs, -2
    else:
        # Ragged M: only A is mapped.
        src_a, contract_a, a_offs, a_ragged_dim = a, -1, offs, -2
        b_offs, b_ragged_dim = None, None

    aq = sa = bq = sb = None
    if is_quantized(a_fmt):
        aq, sa = quantize_operand(
            src_a,
            contract_a,
            a_fmt,
            scale_cfg,
            offs=a_offs,
            ragged_dim=a_ragged_dim,
            stochastic_rounding=a_stochastic_rounding,
            rotation=rotation,
        )
        record_operand(
            a_stats,
            src_a,
            aq,
            sa,
            contract_a,
            scale_cfg,
            offs=a_offs,
            ragged_dim=a_ragged_dim,
            rotation=rotation,
        )
    if is_quantized(b_fmt):
        bq, sb = quantize_operand(
            b,
            -2,
            b_fmt,
            scale_cfg,
            offs=b_offs,
            ragged_dim=b_ragged_dim,
            stochastic_rounding=b_stochastic_rounding,
            rotation=rotation,
        )
        # Keep `offs` for per-expert B metrics, even when B is not ragged.
        record_operand(
            b_stats,
            b,
            bq,
            sb,
            -2,
            scale_cfg,
            offs=offs,
            ragged_dim=b_ragged_dim,
            rotation=rotation,
        )

    if op is not None:
        return SCALED_MM_OPS[op](
            aq.mT if ragged_k else aq,
            bq,
            sa.mT if ragged_k else sa,
            sb,
            offs,
            out_dtype,
            block_size,
            bias=bias,
        )

    if aq is not None:
        src_a = dequantize_operand(
            aq,
            sa,
            contract_a,
            scale_cfg,
            offs=a_offs,
            ragged_dim=a_ragged_dim,
            rotation=rotation,
        ).to(a.dtype)
    if bq is not None:
        b = dequantize_operand(
            bq,
            sb,
            -2,
            scale_cfg,
            offs=b_offs,
            ragged_dim=b_ragged_dim,
            rotation=rotation,
        ).to(b.dtype)
    y = grouped_mm(
        (src_a.mT if ragged_k else src_a).to(out_dtype),
        b.to(out_dtype),
        offs,
        bias=None if bias is None else bias.to(out_dtype),
    )
    return y.to(out_dtype)


class ScaledGroupedGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, bias, offs, cfg: QuantizationConfig, stats, rotation):
        out_dtype = a.dtype
        y = quantized_grouped_mm(
            a,
            b,
            offs,
            cfg.dtype["act"]["fwd"],
            cfg.dtype["weight"]["fwd"],
            out_dtype,
            cfg.scale,
            bias=bias,
            a_stochastic_rounding=cfg.rounding["act"] == "SR",
            b_stochastic_rounding=cfg.rounding["weight"] == "SR",
            a_stats=stats.get("act"),
            b_stats=stats.get("weight"),
            rotation=(
                rotation
                if cfg.rotation is not None and "fwd" in cfg.rotation["gemms"]
                else None
            ),
        )
        ctx.save_for_backward(a, b, offs)
        ctx.cfg = cfg
        ctx.stats = stats
        ctx.rotation = rotation
        ctx.bias_needs_grad = ctx.needs_input_grad[2]
        return y

    @staticmethod
    def backward(ctx, grad_y):
        a, b, offs = ctx.saved_tensors
        cfg = ctx.cfg
        stats = ctx.stats
        out_dtype = a.dtype
        # dgrad: grad_a = grad_y @ b^T
        grad_a = quantized_grouped_mm(
            grad_y,
            b.transpose(-2, -1).contiguous(),
            offs,
            cfg.dtype["grad_out"]["dgrad"],
            cfg.dtype["weight"]["dgrad"],
            out_dtype,
            cfg.scale,
            a_stochastic_rounding=cfg.rounding["grad_out"] == "SR",
            b_stochastic_rounding=cfg.rounding["weight"] == "SR",
            a_stats=stats.get("grad_out"),
            b_stats=stats.get("weight"),
            rotation=(
                ctx.rotation
                if cfg.rotation is not None and "dgrad" in cfg.rotation["gemms"]
                else None
            ),
        )
        # Wgrad uses the ragged-K layout.
        grad_b = quantized_grouped_mm(
            a.mT,
            grad_y,
            offs,
            cfg.dtype["act"]["wgrad"],
            cfg.dtype["grad_out"]["wgrad"],
            out_dtype,
            cfg.scale,
            a_stochastic_rounding=cfg.rounding["act"] == "SR",
            b_stochastic_rounding=cfg.rounding["grad_out"] == "SR",
            a_stats=stats.get("act"),
            b_stats=stats.get("grad_out"),
            rotation=(
                ctx.rotation
                if cfg.rotation is not None and "wgrad" in cfg.rotation["gemms"]
                else None
            ),
        )
        grad_bias = None
        if ctx.bias_needs_grad:
            # Sum each expert's rows in fp32 to avoid bf16 reduction error.
            rows = torch.arange(grad_y.shape[0], device=offs.device)
            group_of_row = torch.searchsorted(offs, rows, right=True)
            acc = grad_y.new_zeros(offs.shape[0], grad_y.shape[1], dtype=torch.float32)
            acc.index_add_(0, group_of_row, grad_y.float())
            grad_bias = acc.to(grad_y.dtype)
        return grad_a, grad_b, grad_bias, None, None, None, None


class QuantizedSparseMoEBlock(SparseMoEBlock):
    @classmethod
    def from_module(
        cls,
        module: SparseMoEBlock,
        quantization_config: QuantizationConfig,
        rotation: Rotation | None = None,
    ) -> "QuantizedSparseMoEBlock":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        object.__setattr__(q, "rotation", rotation)
        # Monitoring populates this; empty disables statistics.
        q.quant_stats = {}
        return q

    def expert_mm(self, a, b, offs, bias=None, projection=None):
        """Quantize training expert GEMMs; use the base block in eval."""
        if not self.training:
            return super().expert_mm(a, b, offs, bias=bias, projection=projection)
        return ScaledGroupedGemmFn.apply(
            a,
            b,
            bias,
            offs,
            self.quantization_config,
            self.quant_stats.get(projection, {}),
            self.rotation,
        )
