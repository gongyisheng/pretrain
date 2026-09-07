import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernel.ops.gemm import SCALED_MM_OPS
from src.metrics.quant import QuantizationStats, record_operand
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.rotation import Rotation
from src.quant.utils import is_quantized, scaled_mm_op
from src.utils.config import QuantizationConfig


def quantized_mm(
    a: torch.Tensor,
    b: torch.Tensor,
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
    """Quantized 2D GEMM with optional per-operand quantization statistics.

    Bias is added before casting to `out_dtype`. `rotation` preconditions both
    operands with the same block-Hadamard; scaled kernels consume those codes
    directly, while emulation inverts each dequantized operand.
    """
    block_shape = scale_cfg.get("block_shape", (0, 0))
    op = scaled_mm_op(
        a_fmt,
        b_fmt,
        scale_cfg.get("scale_dtype"),
        block_shape,
    )
    aq = sa = bq = sb = None
    if is_quantized(a_fmt):
        aq, sa = quantize_operand(
            a,
            -1,
            a_fmt,
            scale_cfg,
            stochastic_rounding=a_stochastic_rounding,
            rotation=rotation,
        )
        record_operand(a_stats, a, aq, sa, -1, scale_cfg, rotation=rotation)
    if is_quantized(b_fmt):
        bq, sb = quantize_operand(
            b,
            -2,
            b_fmt,
            scale_cfg,
            stochastic_rounding=b_stochastic_rounding,
            rotation=rotation,
        )
        record_operand(b_stats, b, bq, sb, -2, scale_cfg, rotation=rotation)

    if op is not None:
        return SCALED_MM_OPS[op](aq, bq, sa, sb, out_dtype, block_shape[1], bias=bias)

    if aq is not None:
        a = dequantize_operand(aq, sa, -1, scale_cfg, rotation=rotation).to(a.dtype)
    if bq is not None:
        b = dequantize_operand(bq, sb, -2, scale_cfg, rotation=rotation).to(b.dtype)
    # Match fused kernels by adding bias in the accumulator before casting.
    y = a @ b if bias is None else torch.addmm(bias, a, b)
    return y.to(out_dtype)


class QuantizedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cfg: QuantizationConfig, stats, rotation):
        # Compute in the active autocast dtype if any, else x's dtype.
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            compute_dtype = torch.get_autocast_dtype(device_type)
        else:
            compute_dtype = x.dtype

        x2d = x.reshape(-1, x.shape[-1]).to(compute_dtype)
        w = weight.to(compute_dtype)
        y = quantized_mm(
            x2d,
            w.t(),
            cfg.dtype["act"]["fwd"],
            cfg.dtype["weight"]["fwd"],
            compute_dtype,
            cfg.scale,
            bias=None if bias is None else bias.to(compute_dtype),
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

        ctx.save_for_backward(x2d, w)
        ctx.cfg = cfg
        # Stats are metadata, not saved tensors.
        ctx.stats = stats
        ctx.rotation = rotation
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        ctx.w_dtype = weight.dtype
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x2d, w = ctx.saved_tensors
        cfg = ctx.cfg
        stats = ctx.stats
        compute_dtype = x2d.dtype
        g = grad_out.reshape(-1, grad_out.shape[-1]).to(compute_dtype)  # (M, N)

        # dX = g @ W, (M,N)@(N,K) -> (M,K)
        dx = quantized_mm(
            g,
            w,
            cfg.dtype["grad_out"]["dgrad"],
            cfg.dtype["weight"]["dgrad"],
            compute_dtype,
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
        # dW = gᵀ @ X, (N,M)@(M,K) -> (N,K)
        dw = quantized_mm(
            g.t(),
            x2d,
            cfg.dtype["grad_out"]["wgrad"],
            cfg.dtype["act"]["wgrad"],
            compute_dtype,
            cfg.scale,
            a_stochastic_rounding=cfg.rounding["grad_out"] == "SR",
            b_stochastic_rounding=cfg.rounding["act"] == "SR",
            a_stats=stats.get("grad_out"),
            b_stats=stats.get("act"),
            rotation=(
                ctx.rotation
                if cfg.rotation is not None and "wgrad" in cfg.rotation["gemms"]
                else None
            ),
        )
        db = g.sum(dim=0) if ctx.has_bias else None

        dx = dx.reshape(*ctx.x_shape).to(ctx.x_dtype)
        dw = dw.to(ctx.w_dtype)
        db = db.to(ctx.w_dtype) if db is not None else None
        return dx, dw, db, None, None, None


class QuantizedLinear(nn.Linear):
    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        quantization_config: QuantizationConfig,
        rotation: Rotation | None = None,
    ) -> "QuantizedLinear":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        object.__setattr__(q, "rotation", rotation)
        # Monitoring attaches stats later; an empty dict disables recording.
        q.quant_stats = {}
        return q

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.weight, self.bias)
        return QuantizedLinearFn.apply(
            x,
            self.weight,
            self.bias,
            self.quantization_config,
            self.quant_stats,
            self.rotation,
        )
