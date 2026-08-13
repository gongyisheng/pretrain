from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.kernel.gemm import scaled_gemm
from src.metrics.quant import record_operand
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.utils import is_fp8, is_int8s, is_quantized
from src.utils.config import QuantizationConfig


def quantized_gemm(
    a, b, a_fmt, b_fmt, out_dtype, scaling_cfg, bias=None, a_stats=None, b_stats=None
):
    """Quantized 2D GEMM. `a_stats`/`b_stats` are optional metric accumulators; when
    armed, each operand's quantization error is folded into its own here, where the
    quantization actually happens, so the measured quantization *is* the performed one
    rather than a replay that has to mirror this path by hand.

    `bias` is an optional (N,) tensor added to the fp32 accumulator in the kernel
    epilogue, so it never passes through the scales. Both paths add it before the
    downcast to `out_dtype`, matching what cuBLAS does for an unquantized addmm.
    """
    same_family = (is_fp8(a_fmt) and is_fp8(b_fmt)) or (
        is_int8s(a_fmt) and is_int8s(b_fmt)
    )

    aq = sa = bq = sb = None
    if is_quantized(a_fmt):
        aq, sa = quantize_operand(a, -1, a_fmt, scaling_cfg)
        record_operand(a_stats, a, aq, sa, -1, scaling_cfg)
    if is_quantized(b_fmt):
        bq, sb = quantize_operand(b, -2, b_fmt, scaling_cfg)
        record_operand(b_stats, b, bq, sb, -2, scaling_cfg)

    if same_family and a.is_cuda and b.is_cuda:
        return scaled_gemm(
            aq,
            bq,
            sa,
            sb,
            out_dtype,
            scaling_cfg["block_size"],
            bias=bias,
            scale_dtype=scaling_cfg.get("scale_dtype"),
        )

    if aq is not None:
        a = dequantize_operand(aq, sa, -1, scaling_cfg).to(a.dtype)
    if bq is not None:
        b = dequantize_operand(bq, sb, -2, scaling_cfg).to(b.dtype)
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
        y = quantized_gemm(
            x2d,
            w.t(),
            cfg.dtype["act"],
            cfg.dtype["weight"],
            compute_dtype,
            cfg.scaling,
            bias=None if bias is None else bias.to(compute_dtype),
            a_stats=stats.get("fwd_act"),
            b_stats=stats.get("fwd_weight"),
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
        dx = quantized_gemm(
            g,
            w,
            cfg.dtype["grad_input"],
            cfg.dtype["weight"],
            compute_dtype,
            cfg.scaling,
            a_stats=stats.get("dgrad_grad"),
            b_stats=stats.get("dgrad_weight"),
        )
        # dW = gᵀ @ X, (N,M)@(M,K) -> (N,K)
        dw = quantized_gemm(
            g.t(),
            x2d,
            cfg.dtype["grad_weight"],
            cfg.dtype["act"],
            compute_dtype,
            cfg.scaling,
            a_stats=stats.get("wgrad_grad"),
            b_stats=stats.get("wgrad_act"),
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
        return QuantizedLinearFn.apply(
            x, self.weight, self.bias, self.quantization_config, self.quant_stats
        )
