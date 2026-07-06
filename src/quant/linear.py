from __future__ import annotations

import torch
import torch.nn as nn

from src.quant.fp8 import fp8_gemm, fake_quantize_fp8
from src.quant.int8 import int8_gemm, fake_quantize_int8
from src.quant.utils import (
    str_to_dtype_fp8,
    str_to_qmax_int8s,
    is_fp8,
    is_int8s,
    is_quantized,
)
from src.utils.config import QuantConfig


def _fake_quant(x, fmt, dim):
    """dequant(quant(x)) for one operand in its format; identity if passthrough."""
    if is_fp8(fmt):
        return fake_quantize_fp8(x, str_to_dtype_fp8(fmt), dim=dim)
    if is_int8s(fmt):
        return fake_quantize_int8(x, str_to_qmax_int8s(fmt), dim=dim)
    return x


def _gemm(a, b, a_fmt, b_fmt, out_dtype, rowwise=False):
    """`a @ b` dispatched by operand format family.

    Both operands the same quantized family -> real fused GEMM. Otherwise
    (mixed families, one-sided, or passthrough) fake-quant each quantized
    operand and matmul in high precision.
    """
    if is_fp8(a_fmt) and is_fp8(b_fmt):
        return fp8_gemm(
            a,
            b,
            out_dtype,
            str_to_dtype_fp8(a_fmt),
            str_to_dtype_fp8(b_fmt),
            rowwise=rowwise,
        )
    if is_int8s(a_fmt) and is_int8s(b_fmt):
        return int8_gemm(
            a,
            b,
            out_dtype,
            str_to_qmax_int8s(a_fmt),
            str_to_qmax_int8s(b_fmt),
            rowwise=rowwise,
        )
    if is_quantized(a_fmt):
        a = _fake_quant(a, a_fmt, dim=-1 if rowwise else None)
    if is_quantized(b_fmt):
        b = _fake_quant(b, b_fmt, dim=0 if rowwise else None)
    return (a @ b).to(out_dtype)


class QuantizedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cfg: QuantConfig):
        # Follow the active autocast dtype so operands share a dtype and the
        # output matches what nn.Linear would produce; else use x's dtype.
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            compute_dtype = torch.get_autocast_dtype(device_type)
        else:
            compute_dtype = x.dtype

        rowwise = cfg.scaling["granularity"] == "rowwise"

        x2d = x.reshape(-1, x.shape[-1]).to(compute_dtype)
        w = weight.to(compute_dtype)
        y = _gemm(
            x2d, w.t(), cfg.dtype["act"], cfg.dtype["weight"], compute_dtype, rowwise
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

        rowwise = cfg.scaling["granularity"] == "rowwise"

        # dX = g @ W          (M,N)@(N,K) -> (M,K)
        dx = _gemm(
            g, w, cfg.dtype["input_grad"], cfg.dtype["weight"], compute_dtype, rowwise
        )
        # dW = gᵀ @ X         (N,M)@(M,K) -> (N,K)
        dw = _gemm(
            g.t(),
            x2d,
            cfg.dtype["weight_grad"],
            cfg.dtype["act"],
            compute_dtype,
            rowwise,
        )
        db = g.sum(dim=0) if ctx.has_bias else None

        dx = dx.reshape(*ctx.x_shape).to(ctx.x_dtype)
        dw = dw.to(ctx.w_dtype)
        db = db.to(ctx.w_dtype) if db is not None else None
        return dx, dw, db, None


class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, cfg: QuantConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, cfg: QuantConfig) -> "QuantLinear":
        q = cls(linear.in_features, linear.out_features, linear.bias is not None, cfg)
        with torch.no_grad():
            q.weight.copy_(linear.weight)
            if linear.bias is not None:
                q.bias.copy_(linear.bias)
        return q.to(linear.weight.device, linear.weight.dtype)

    def forward(self, x):
        return QuantizedLinearFn.apply(x, self.weight, self.bias, self.cfg)
