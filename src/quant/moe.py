from __future__ import annotations

import torch

from src.kernel.gemm import (
    grouped_gemm,
    scaled_grouped_gemm,
    scaled_grouped_gemm_wgrad,
)
from src.quant.quantize import (
    effective_block_size,
    fake_quantize_operand,
    quantize_operand,
)
from src.quant.utils import is_fp8, is_int8s, is_quantized
from src.utils.config import QuantConfig


def _same_family(a_fmt: str, b_fmt: str) -> bool:
    return (is_fp8(a_fmt) and is_fp8(b_fmt)) or (is_int8s(a_fmt) and is_int8s(b_fmt))


def _quant_a(a, fmt, scaling):
    gran = scaling["granularity"]
    bs = scaling.get("block_size", 0)
    return quantize_operand(
        a, -1, gran, bs, fmt, scale_dtype=scaling.get("scale_dtype")
    )


def _quant_b(b, fmt, scaling):
    """Per-expert quantize (E,K,N) along K (dim 0 of each slab)."""
    gran = scaling["granularity"]
    bs = scaling.get("block_size", 0)
    sd = scaling.get("scale_dtype")
    qs = [
        quantize_operand(b[g], 0, gran, bs, fmt, scale_dtype=sd)
        for g in range(b.shape[0])
    ]
    bq = torch.stack([q for q, _ in qs])
    sb = torch.stack([s for _, s in qs])
    return bq, sb


def _grouped(a, b, offs, a_fmt, b_fmt, out_dtype, scaling):
    """Quantized ragged (a @ b[g]); fake-quant + bf16 fallback off the fused path."""
    gran = scaling["granularity"]
    bs = scaling.get("block_size", 0)
    fused = _same_family(a_fmt, b_fmt) and a.is_cuda and gran != "blockwise"
    if fused:
        ebs = effective_block_size(gran, bs, a.shape[1])
        aq, sa = _quant_a(a, a_fmt, scaling)
        bq, sb = _quant_b(b, b_fmt, scaling)
        return scaled_grouped_gemm(aq, bq, sa, sb, offs, out_dtype, ebs)
    a_fq = fake_quantize_operand(a, -1, gran, bs, a_fmt) if is_quantized(a_fmt) else a
    b_fq = b
    if is_quantized(b_fmt):
        b_fq = torch.stack(
            [fake_quantize_operand(b[g], 0, gran, bs, b_fmt) for g in range(b.shape[0])]
        )
    return grouped_gemm(a_fq.to(out_dtype), b_fq.to(out_dtype), offs).to(out_dtype)


def _wgrad(a, g, offs, a_fmt, g_fmt, out_dtype, scaling):
    """grad_b[g] = (a·sa)[g]^T @ (g·sg)[g] -> (E,K,N). fp8/int8 rowwise|tensorwise
    use the fused kernel; blockwise falls back to fake-quant + bf16 wgrad."""
    gran = scaling["granularity"]
    bs = scaling.get("block_size", 0)
    fused = _same_family(a_fmt, g_fmt) and a.is_cuda and gran != "blockwise"
    if fused:
        aq, sa = quantize_operand(a, 0, gran, bs, a_fmt)  # sa (1,K)
        gq, sg = quantize_operand(g, 0, gran, bs, g_fmt)  # sg (1,N)
        return scaled_grouped_gemm_wgrad(aq, gq, sa, sg, offs, out_dtype)
    a_fq = fake_quantize_operand(a, 0, gran, bs, a_fmt) if is_quantized(a_fmt) else a
    g_fq = fake_quantize_operand(g, 0, gran, bs, g_fmt) if is_quantized(g_fmt) else g
    # bf16 ragged Xᵀ@gY via the existing grouped wgrad path
    from src.kernel.gemm import _grouped_gemm_wgrad

    return _grouped_gemm_wgrad(a_fq.to(out_dtype), g_fq.to(out_dtype), offs).to(
        out_dtype
    )


class ScaledGroupedGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offs, cfg: QuantConfig):
        out_dtype = a.dtype
        y = _grouped(
            a, b, offs, cfg.dtype["act"], cfg.dtype["weight"], out_dtype, cfg.scaling
        )
        ctx.save_for_backward(a, b, offs)
        ctx.cfg = cfg
        return y

    @staticmethod
    def backward(ctx, grad_y):
        a, b, offs = ctx.saved_tensors
        cfg = ctx.cfg
        out_dtype = a.dtype
        # dgrad: grad_a = grad_y @ b^T  (contract over N); reuse the forward op.
        grad_a = _grouped(
            grad_y,
            b.transpose(-2, -1).contiguous(),
            offs,
            cfg.dtype["input_grad"],
            cfg.dtype["weight"],
            out_dtype,
            cfg.scaling,
        )
        # wgrad: grad_b[g] = a[g]^T @ grad_y[g]  (contract over ragged M).
        grad_b = _wgrad(
            a,
            grad_y,
            offs,
            cfg.dtype["act"],
            cfg.dtype["weight_grad"],
            out_dtype,
            cfg.scaling,
        )
        return grad_a, grad_b, None, None


def quantized_expert_mm(cfg: QuantConfig):
    """Build an expert_mm bound to `cfg`: a differentiable quantized ragged grouped
    GEMM (a (R,K), b (E,K,N), offs (E,)) -> (R,N) that quantizes operands per cfg
    before the fused kernel. Same (a, b, offs) call contract as `grouped_gemm`, so
    it drops into `grouped_mlp`/`SparseMoEBlock.expert_mm`.
    """

    def expert_mm(a, b, offs):
        return ScaledGroupedGemmFn.apply(a, b, offs, cfg)

    return expert_mm
