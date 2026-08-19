from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernel.ops.gemm import SCALED_MM_OPS
from src.metrics.quant import QuantizationStats, record_operand
from src.quant.quantize import dequantize_operand, quantize_operand
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
) -> torch.Tensor:
    """Quantized 2D GEMM. `a_stats`/`b_stats` are optional metric accumulators; when
    armed, each operand's quantization error is folded into its own here, where the
    quantization actually happens, so the measured quantization *is* the performed one
    rather than a replay that has to mirror this path by hand.

    `bias` is an optional (N,) tensor added to the fp32 accumulator in the kernel
    epilogue, so it never passes through the scales. Both paths add it before the
    downcast to `out_dtype`, matching what cuBLAS does for an unquantized addmm.
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
            a, -1, a_fmt, scale_cfg, stochastic_rounding=a_stochastic_rounding
        )
        record_operand(a_stats, a, aq, sa, -1, scale_cfg)
    if is_quantized(b_fmt):
        bq, sb = quantize_operand(
            b, -2, b_fmt, scale_cfg, stochastic_rounding=b_stochastic_rounding
        )
        record_operand(b_stats, b, bq, sb, -2, scale_cfg)

    if op is not None:
        return SCALED_MM_OPS[op](aq, bq, sa, sb, out_dtype, block_shape[1], bias=bias)

    if aq is not None:
        a = dequantize_operand(aq, sa, -1, scale_cfg).to(a.dtype)
    if bq is not None:
        b = dequantize_operand(bq, sb, -2, scale_cfg).to(b.dtype)
    y = a @ b
    if bias is not None:
        y = y + bias
    return y.to(out_dtype)


class QuantizedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cfg: QuantizationConfig, stats):
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
        )

        ctx.save_for_backward(x2d, w)
        ctx.cfg = cfg
        # a plain dict of modules, carried the same way cfg is -- not a saved tensor
        ctx.stats = stats
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
        )
        db = g.sum(dim=0) if ctx.has_bias else None

        dx = dx.reshape(*ctx.x_shape).to(ctx.x_dtype)
        dw = dw.to(ctx.w_dtype)
        db = db.to(ctx.w_dtype) if db is not None else None
        return dx, dw, db, None, None


class QuantizedLinear(nn.Linear):
    @classmethod
    def from_module(
        cls, module: nn.Linear, quantization_config: QuantizationConfig
    ) -> "QuantizedLinear":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        # populated by src.metrics.convert.apply_quantization_monitoring; empty means
        # every .get() below returns None and the folds are skipped outright
        q.quant_stats = {}
        return q

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.weight, self.bias)
        return QuantizedLinearFn.apply(
            x, self.weight, self.bias, self.quantization_config, self.quant_stats
        )
