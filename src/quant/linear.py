from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.kernel.gemm import scaled_gemm
from src.quant.quantize import (
    QuantizationSnapshot,
    dequantize_operand,
    effective_block_size,
    quantize_operand,
)
from src.quant.utils import is_fp8, is_int8s, is_quantized
from src.utils.config import QuantConfig


def _quant_gemm(a, b, a_fmt, b_fmt, out_dtype, scaling_cfg):
    """`a @ b` via a fused GEMM when both operands share a quantized family, else fake-quant.

    Also returns a QuantizationSnapshot per operand (None if its fmt is passthrough);
    a fused dispatch implies both are quantized, so both are set on that path.
    """
    granularity = scaling_cfg.get("granularity", "tensorwise")
    block_size = scaling_cfg.get("block_size", 0)
    scale_dtype = scaling_cfg.get("scale_dtype")
    same_family = (is_fp8(a_fmt) and is_fp8(b_fmt)) or (
        is_int8s(a_fmt) and is_int8s(b_fmt)
    )

    a_snap = b_snap = None
    if is_quantized(a_fmt):
        aq, sa = quantize_operand(
            a, -1, granularity, block_size, a_fmt, scale_dtype=scale_dtype
        )
        a_snap = QuantizationSnapshot(a, aq, sa, -1, granularity, block_size)
    if is_quantized(b_fmt):
        bq, sb = quantize_operand(
            b, 0, granularity, block_size, b_fmt, scale_dtype=scale_dtype
        )
        b_snap = QuantizationSnapshot(b, bq, sb, 0, granularity, block_size)

    if same_family and a.is_cuda and b.is_cuda:
        y = scaled_gemm(
            a_snap.quantized_tensor,
            b_snap.quantized_tensor,
            a_snap.scale,
            b_snap.scale,
            out_dtype,
            effective_block_size(granularity, block_size, a.shape[1]),
        )
        return y, a_snap, b_snap

    if a_snap is not None:
        a = dequantize_operand(aq, sa, -1, granularity, block_size).to(a.dtype)
    if b_snap is not None:
        b = dequantize_operand(bq, sb, 0, granularity, block_size).to(b.dtype)
    return (a @ b).to(out_dtype), a_snap, b_snap


class QuantizedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cfg: QuantConfig, probe=None):
        # Compute in the active autocast dtype if any, else x's dtype.
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            compute_dtype = torch.get_autocast_dtype(device_type)
        else:
            compute_dtype = x.dtype

        x2d = x.reshape(-1, x.shape[-1]).to(compute_dtype)
        w = weight.to(compute_dtype)
        y, act_snap, weight_snap = _quant_gemm(
            x2d,
            w.t(),
            cfg.dtype["act"],
            cfg.dtype["weight"],
            compute_dtype,
            cfg.scaling,
        )
        if probe is not None and probe.enabled:
            probe.record("fwd.act", act_snap)
            probe.record("fwd.weight", weight_snap)
        if bias is not None:
            y = y + bias.to(compute_dtype)

        ctx.save_for_backward(x2d, w)
        ctx.cfg = cfg
        ctx.probe = probe
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
        dx, dgrad_grad_snap, dgrad_weight_snap = _quant_gemm(
            g,
            w,
            cfg.dtype["grad_input"],
            cfg.dtype["weight"],
            compute_dtype,
            cfg.scaling,
        )
        # dW = gᵀ @ X, (N,M)@(M,K) -> (N,K)
        dw, wgrad_grad_snap, wgrad_act_snap = _quant_gemm(
            g.t(),
            x2d,
            cfg.dtype["grad_weight"],
            cfg.dtype["act"],
            compute_dtype,
            cfg.scaling,
        )
        probe = ctx.probe
        if probe is not None and probe.enabled:
            probe.record("dgrad.grad", dgrad_grad_snap)
            probe.record("dgrad.weight", dgrad_weight_snap)
            probe.record("wgrad.grad", wgrad_grad_snap)
            probe.record("wgrad.act", wgrad_act_snap)
        db = g.sum(dim=0) if ctx.has_bias else None

        dx = dx.reshape(*ctx.x_shape).to(ctx.x_dtype)
        dw = dw.to(ctx.w_dtype)
        db = db.to(ctx.w_dtype) if db is not None else None
        return dx, dw, db, None, None


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        quantization_config: QuantConfig = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.quantization_config = quantization_config
        self.quantization_probe = None  # set by attach_quant_probe when logging metrics

    @classmethod
    def from_module(
        cls, module: nn.Linear, quantization_config: QuantConfig
    ) -> "QuantizedLinear":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        q.quantization_probe = None
        return q

    def forward(self, x):
        return QuantizedLinearFn.apply(
            x, self.weight, self.bias, self.quantization_config, self.quantization_probe
        )
