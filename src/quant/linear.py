from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.quant.utils import (
    is_fp8,
    is_int8s,
    is_quantized,
)
from src.quant.quantize import (
    quantize_operand,
    fake_quantize_operand,
    effective_block_size,
)
from src.kernel.gemm import scaled_gemm
from src.utils.config import QuantConfig


def _quant_gemm(a, b, a_fmt, b_fmt, out_dtype, scaling_cfg):
    """`a @ b` via a fused GEMM when both operands share a quantized family, else fake-quant."""
    granularity = scaling_cfg.get("granularity", "tensorwise")
    block_size = scaling_cfg.get("block_size", 0)
    scale_dtype = scaling_cfg.get("scale_dtype")
    same_family = (is_fp8(a_fmt) and is_fp8(b_fmt)) or (
        is_int8s(a_fmt) and is_int8s(b_fmt)
    )
    if same_family and a.is_cuda and b.is_cuda:
        ebs = effective_block_size(granularity, block_size, a.shape[1])
        aq, sa = quantize_operand(
            a, -1, granularity, block_size, a_fmt, scale_dtype=scale_dtype
        )
        bq, sb = quantize_operand(
            b, 0, granularity, block_size, b_fmt, scale_dtype=scale_dtype
        )
        return scaled_gemm(aq, bq, sa, sb, out_dtype, ebs)
    if is_quantized(a_fmt):
        a = fake_quantize_operand(
            a, -1, granularity, block_size, a_fmt, scale_dtype=scale_dtype
        )
    if is_quantized(b_fmt):
        b = fake_quantize_operand(
            b, 0, granularity, block_size, b_fmt, scale_dtype=scale_dtype
        )
    return (a @ b).to(out_dtype)


class QuantizedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cfg: QuantConfig):
        # Compute in the active autocast dtype if any, else x's dtype.
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            compute_dtype = torch.get_autocast_dtype(device_type)
        else:
            compute_dtype = x.dtype

        x2d = x.reshape(-1, x.shape[-1]).to(compute_dtype)
        w = weight.to(compute_dtype)
        y = _quant_gemm(
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
        dx = _quant_gemm(
            g,
            w,
            cfg.dtype["grad_input"],
            cfg.dtype["weight"],
            compute_dtype,
            cfg.scaling,
        )
        # dW = gᵀ @ X, (N,M)@(M,K) -> (N,K)
        dw = _quant_gemm(
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
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        quantization_config: QuantConfig = None,
        layer_id=None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.quantization_config = quantization_config
        self.layer_id = layer_id

    @classmethod
    def from_module(
        cls, module: nn.Linear, quantization_config: QuantConfig, layer_id=None
    ) -> "QuantizedLinear":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        q.layer_id = layer_id
        return q

    def forward(self, x):
        return QuantizedLinearFn.apply(
            x, self.weight, self.bias, self.quantization_config
        )
