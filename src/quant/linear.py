from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.kernel.gemm import scaled_gemm
from src.quant.quantize import (
    QuantizationSnapshot,
    dequantize_operand,
    quantize_operand,
)
from src.quant.utils import is_fp8, is_int8s, is_quantized
from src.utils.config import QuantizationConfig


def quantized_gemm(a, b, a_fmt, b_fmt, out_dtype, scaling_cfg, enable_snapshot=False):
    """Returns (y, a_snap, b_snap). The snapshots are the diagnostic replay's only
    product and are `None` unless `enable_snapshot`; training discards them, so the
    default keeps the operand tensors off the hot path. Kept a parameter rather than a
    module-level flag: the compiled path only ever passes False, so Dynamo sees one
    value, whereas a flag that flips would specialize and recompile.
    """
    granularity = scaling_cfg.get("granularity", "tensorwise")
    block_size = scaling_cfg.get("block_size", 0)
    scale_dtype = scaling_cfg.get("scale_dtype")
    same_family = (is_fp8(a_fmt) and is_fp8(b_fmt)) or (
        is_int8s(a_fmt) and is_int8s(b_fmt)
    )

    a_snap = b_snap = None
    aq = sa = bq = sb = None
    if is_quantized(a_fmt):
        aq, sa = quantize_operand(
            a, -1, granularity, block_size, a_fmt, scale_dtype=scale_dtype
        )
        if enable_snapshot:
            a_snap = QuantizationSnapshot(a, aq, sa, -1, block_size, fmt=a_fmt)
    if is_quantized(b_fmt):
        bq, sb = quantize_operand(
            b, 0, granularity, block_size, b_fmt, scale_dtype=scale_dtype
        )
        if enable_snapshot:
            b_snap = QuantizationSnapshot(b, bq, sb, 0, block_size, fmt=b_fmt)

    if same_family and a.is_cuda and b.is_cuda:
        y = scaled_gemm(aq, bq, sa, sb, out_dtype, block_size)
        return y, a_snap, b_snap

    if aq is not None:
        a = dequantize_operand(aq, sa, -1, block_size).to(a.dtype)
    if bq is not None:
        b = dequantize_operand(bq, sb, 0, block_size).to(b.dtype)
    return (a @ b).to(out_dtype), a_snap, b_snap


class QuantizedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cfg: QuantizationConfig):
        # Compute in the active autocast dtype if any, else x's dtype.
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            compute_dtype = torch.get_autocast_dtype(device_type)
        else:
            compute_dtype = x.dtype

        x2d = x.reshape(-1, x.shape[-1]).to(compute_dtype)
        w = weight.to(compute_dtype)
        y, _, _ = quantized_gemm(
            x2d,
            w.t(),
            cfg.dtype["act"],
            cfg.dtype["weight"],
            compute_dtype,
            cfg.scaling,
        )
        if bias is not None:
            y = y + bias.to(compute_dtype)

        ctx.save_for_backward(x2d, w)
        ctx.cfg = cfg
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        ctx.w_dtype = weight.dtype
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x2d, w = ctx.saved_tensors
        cfg = ctx.cfg
        compute_dtype = x2d.dtype
        g = grad_out.reshape(-1, grad_out.shape[-1]).to(compute_dtype)  # (M, N)

        # dX = g @ W, (M,N)@(N,K) -> (M,K)
        dx, _, _ = quantized_gemm(
            g,
            w,
            cfg.dtype["grad_input"],
            cfg.dtype["weight"],
            compute_dtype,
            cfg.scaling,
        )
        # dW = gᵀ @ X, (N,M)@(M,K) -> (N,K)
        dw, _, _ = quantized_gemm(
            g.t(),
            x2d,
            cfg.dtype["grad_weight"],
            cfg.dtype["act"],
            compute_dtype,
            cfg.scaling,
        )
        db = g.sum(dim=0) if ctx.has_bias else None

        dx = dx.reshape(*ctx.x_shape).to(ctx.x_dtype)
        dw = dw.to(ctx.w_dtype)
        db = db.to(ctx.w_dtype) if db is not None else None
        return dx, dw, db, None


class QuantizedLinear(nn.Linear):
    @classmethod
    def from_module(
        cls, module: nn.Linear, quantization_config: QuantizationConfig
    ) -> "QuantizedLinear":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        return q

    def forward(self, x):
        return QuantizedLinearFn.apply(
            x, self.weight, self.bias, self.quantization_config
        )
