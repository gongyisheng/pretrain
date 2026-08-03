from __future__ import annotations

import copy

import torch

from src.kernel.gemm import (
    grouped_gemm,
    grouped_gemm_wgrad,
    scaled_grouped_gemm,
    scaled_grouped_gemm_wgrad,
)
from src.layers.mlp import SparseMoEBlock
from src.quant.quantize import (
    QuantizationSnapshot,
    dequantize_operand,
    effective_block_size,
    quantize_operand,
)
from src.quant.utils import is_fp8, is_int8s, is_quantized
from src.utils.config import QuantizationConfig


def quantized_grouped_gemm(a, b, offs, a_fmt, b_fmt, out_dtype, scaling):
    granularity = scaling["granularity"]
    block_size = scaling.get("block_size", 0)
    scale_dtype = scaling.get("scale_dtype")
    same_family = (is_fp8(a_fmt) and is_fp8(b_fmt)) or (
        is_int8s(a_fmt) and is_int8s(b_fmt)
    )

    a_snap = b_snap = None
    aq = sa = bq = sb = None
    if is_quantized(a_fmt):
        aq, sa = quantize_operand(
            a, -1, granularity, block_size, a_fmt, scale_dtype=scale_dtype
        )
        a_snap = QuantizationSnapshot(a, aq, sa, -1, granularity, block_size, offs)
    if is_quantized(b_fmt):
        per_expert = [
            quantize_operand(
                b[g], 0, granularity, block_size, b_fmt, scale_dtype=scale_dtype
            )
            for g in range(b.shape[0])
        ]
        bq = torch.stack([q for q, _ in per_expert])
        sb = torch.stack([s for _, s in per_expert])
        b_snap = QuantizationSnapshot(b, bq, sb, 1, granularity, block_size, offs)

    if same_family and a.is_cuda:
        y = scaled_grouped_gemm(
            aq,
            bq,
            sa,
            sb,
            offs,
            out_dtype,
            effective_block_size(granularity, block_size, a.shape[1]),
        )
        return y, a_snap, b_snap

    if aq is not None:
        a = dequantize_operand(aq, sa, -1, granularity, block_size).to(a.dtype)
    if bq is not None:
        b = dequantize_operand(bq, sb, 1, granularity, block_size).to(b.dtype)
    y = grouped_gemm(a.to(out_dtype), b.to(out_dtype), offs).to(out_dtype)
    return y, a_snap, b_snap


def quantized_grouped_wgrad(a, g, offs, a_fmt, g_fmt, out_dtype, scaling):
    granularity = scaling["granularity"]
    block_size = scaling.get("block_size", 0)
    scale_dtype = scaling.get("scale_dtype")
    same_family = (is_fp8(a_fmt) and is_fp8(g_fmt)) or (
        is_int8s(a_fmt) and is_int8s(g_fmt)
    )

    a_snap = g_snap = None
    aq = sa = gq = sg = None
    if is_quantized(a_fmt):
        aq, sa = quantize_operand(
            a, 0, granularity, block_size, a_fmt, scale_dtype=scale_dtype
        )
        a_snap = QuantizationSnapshot(
            a, aq, sa, 0, granularity, block_size, offs
        )  # (nrb,K)
    if is_quantized(g_fmt):
        gq, sg = quantize_operand(
            g, 0, granularity, block_size, g_fmt, scale_dtype=scale_dtype
        )
        g_snap = QuantizationSnapshot(
            g, gq, sg, 0, granularity, block_size, offs
        )  # (nrb,N)

    if same_family and a.is_cuda:
        grad_b = scaled_grouped_gemm_wgrad(
            aq,
            gq,
            sa,
            sg,
            offs,
            out_dtype,
            effective_block_size(granularity, block_size, a.shape[0]),
        )
        return grad_b, a_snap, g_snap

    if aq is not None:
        a = dequantize_operand(aq, sa, 0, granularity, block_size).to(a.dtype)
    if gq is not None:
        g = dequantize_operand(gq, sg, 0, granularity, block_size).to(g.dtype)
    # bf16 ragged Xᵀ@gY via the existing grouped wgrad path
    grad_b = grouped_gemm_wgrad(a.to(out_dtype), g.to(out_dtype), offs).to(out_dtype)
    return grad_b, a_snap, g_snap


class ScaledGroupedGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offs, cfg: QuantizationConfig, probe=None):
        out_dtype = a.dtype
        y, act_snap, weight_snap = quantized_grouped_gemm(
            a, b, offs, cfg.dtype["act"], cfg.dtype["weight"], out_dtype, cfg.scaling
        )
        if probe is not None and probe.enabled:
            probe.record("fwd.act", act_snap)
            probe.record("fwd.weight", weight_snap)
        ctx.save_for_backward(a, b, offs)
        ctx.cfg = cfg
        ctx.probe = probe
        return y

    @staticmethod
    def backward(ctx, grad_y):
        a, b, offs = ctx.saved_tensors
        cfg = ctx.cfg
        out_dtype = a.dtype
        # dgrad: grad_a = grad_y @ b^T
        grad_a, dgrad_grad_snap, dgrad_weight_snap = quantized_grouped_gemm(
            grad_y,
            b.transpose(-2, -1).contiguous(),
            offs,
            cfg.dtype["grad_input"],
            cfg.dtype["weight"],
            out_dtype,
            cfg.scaling,
        )
        # wgrad: grad_b[g] = a[g]^T @ grad_y[g]
        grad_b, wgrad_act_snap, wgrad_grad_snap = quantized_grouped_wgrad(
            a,
            grad_y,
            offs,
            cfg.dtype["act"],
            cfg.dtype["grad_weight"],
            out_dtype,
            cfg.scaling,
        )
        probe = ctx.probe
        if probe is not None and probe.enabled:
            probe.record("dgrad.grad", dgrad_grad_snap)
            probe.record("dgrad.weight", dgrad_weight_snap)
            probe.record("wgrad.act", wgrad_act_snap)
            probe.record("wgrad.grad", wgrad_grad_snap)
        return grad_a, grad_b, None, None, None


def quantized_expert_mm(cfg: QuantizationConfig, probes=None):

    def expert_mm(a, b, offs, projection=None):
        probe = probes.get(projection) if probes else None
        return ScaledGroupedGemmFn.apply(a, b, offs, cfg, probe)

    return expert_mm


class QuantizedSparseMoEBlock(SparseMoEBlock):
    @classmethod
    def from_module(
        cls, module: SparseMoEBlock, quantization_config: QuantizationConfig
    ) -> "QuantizedSparseMoEBlock":
        q = cls.__new__(cls)
        q.__dict__ = copy.deepcopy(module).__dict__
        q.quantization_config = quantization_config
        q.expert_mm = quantized_expert_mm(quantization_config)
        return q

    def set_quantization_probe(self, gate_up_probe, down_probe):
        self.expert_mm = quantized_expert_mm(
            self.quantization_config,
            {"gate_up": gate_up_probe, "down": down_probe},
        )
