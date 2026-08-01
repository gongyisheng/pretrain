from __future__ import annotations

import torch

from src.quant.constants import EPS
from src.quant.quantize import operand_quotient
from src.quant.utils import is_quantized, str_to_min_subnormal, str_to_qmax


def compute_operand_metrics(
    x, fmt, granularity, block_size, contract_dim, *, scale_dtype=None
):
    """Per-tensor quant health metrics, or None for a passthrough fmt.

    Returns 0-dim float32 tensors on x.device: underflow_rate, clip_rate,
    range_ratio, sqnr. No host sync (no .item()).
    """
    if not is_quantized(fmt):
        return None

    q_abs, deq, xf = operand_quotient(
        x, contract_dim, granularity, block_size, fmt, scale_dtype=scale_dtype
    )
    qmax = str_to_qmax(fmt)
    min_sub = str_to_min_subnormal(fmt)

    nonzero = q_abs > 0
    total = xf.numel()
    underflow = (nonzero & (q_abs < min_sub)).sum().float() / total
    clip = (q_abs > qmax).sum().float() / total

    absxf = xf.abs()
    amax = absxf.amax()
    median = absxf.median().clamp_min(EPS)
    range_ratio = amax / median

    err = (xf - deq).norm()
    sig = xf.norm().clamp_min(EPS)
    sqnr = 20.0 * torch.log10((sig / err.clamp_min(EPS)))

    return {
        "underflow_rate": underflow,
        "clip_rate": clip,
        "range_ratio": range_ratio,
        "sqnr": sqnr,
    }
