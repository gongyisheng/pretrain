from __future__ import annotations

import torch
import torch.nn as nn

from src.quant.fp8 import fp8_gemm, fake_quantize_fp8
from src.quant.utils import quant_dtype
from src.utils.config import QuantConfig


def _gemm(a, b, a_dtype, b_dtype, out_dtype, rowwise=False):
    if a_dtype is not None and b_dtype is not None:
        return fp8_gemm(a, b, out_dtype, a_dtype, b_dtype, rowwise=rowwise)
    if a_dtype is not None:
        a = fake_quantize_fp8(a, a_dtype, dim=-1 if rowwise else None)
    if b_dtype is not None:
        b = fake_quantize_fp8(b, b_dtype, dim=0 if rowwise else None)
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

        act_dtype = quant_dtype(cfg.dtype.get("act"))
        weight_dtype = quant_dtype(cfg.dtype.get("weight"))
        rowwise = cfg.scaling.get("granularity") == "rowwise"

        x2d = x.reshape(-1, x.shape[-1]).to(compute_dtype)
        w = weight.to(compute_dtype)
        y = _gemm(x2d, w.t(), act_dtype, weight_dtype, compute_dtype, rowwise)
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

        grad_dtype = quant_dtype(cfg.dtype.get("grad"))
        weight_dtype = quant_dtype(cfg.dtype.get("weight"))
        act_dtype = quant_dtype(cfg.dtype.get("act"))
        rowwise = cfg.scaling.get("granularity") == "rowwise"

        # dX = g @ W          (M,N)@(N,K) -> (M,K)
        dx = _gemm(g, w, grad_dtype, weight_dtype, compute_dtype, rowwise)
        # dW = gᵀ @ X         (N,M)@(M,K) -> (N,K)
        dw = _gemm(g.t(), x2d, grad_dtype, act_dtype, compute_dtype, rowwise)
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
