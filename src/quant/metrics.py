from __future__ import annotations

import torch

from src.layers.mlp import SparseMoEBlock
from src.quant.constants import EPS
from src.quant.linear import QuantizedLinear
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


class QuantMetricCollector:
    """Accumulates per-(layer, operand, metric) 0-dim tensors across a step's hooks.

    No host sync in `record`; `drain` performs the single `.tolist()` sync.
    """

    def __init__(self):
        self.enabled = False
        self._store = {}  # (layer_id, operand, metric) -> 0-dim tensor

    def record(self, layer_id, operand, metrics):
        if metrics is None:
            return
        for name, val in metrics.items():
            self._store[(layer_id, operand, name)] = val

    def drain(self):
        if not self._store:
            return {}
        keys = list(self._store.keys())
        vals = torch.stack([self._store[k] for k in keys]).tolist()  # one sync
        self._store = {}
        return {
            f"quant/{metric}/{operand}/layer_{layer_id}": v
            for (layer_id, operand, metric), v in zip(keys, vals)
        }


def _record_operand(collector, module, tensor, operand, fmt_key):
    cfg = module.quantization_config
    fmt = cfg.dtype[fmt_key]
    m = compute_operand_metrics(
        tensor,
        fmt,
        cfg.scaling.get("granularity", "tensorwise"),
        cfg.scaling.get("block_size", 0),
        -1,
        scale_dtype=cfg.scaling.get("scale_dtype"),
    )
    collector.record(module.layer_id, operand, m)


def _make_hooks(collector):
    def fwd(module, inputs, output):
        if not collector.enabled:
            return
        _record_operand(collector, module, module.weight, "weight", "weight")
        _record_operand(collector, module, inputs[0], "act", "act")

    def bwd(module, grad_input, grad_output):
        if not collector.enabled:
            return
        g = grad_output[0]
        _record_operand(collector, module, g, "grad_input", "grad_input")
        _record_operand(collector, module, g, "grad_weight", "grad_weight")

    return fwd, bwd


def _make_moe_hooks(collector):
    def fwd(module, inputs, output):
        if not collector.enabled:
            return
        # MoE act/grad are pre-router (approximation); weight is faithful.
        weight = module.expert_gate_up if module.gated else module.expert_up
        _record_operand(collector, module, weight, "weight", "weight")
        _record_operand(collector, module, inputs[0], "act", "act")

    def bwd(module, grad_input, grad_output):
        if not collector.enabled:
            return
        g = grad_output[0]
        _record_operand(collector, module, g, "grad_input", "grad_input")
        _record_operand(collector, module, g, "grad_weight", "grad_weight")

    return fwd, bwd


def register_quant_metric_hooks(model, collector):
    handles = []
    for module in model.modules():
        if isinstance(module, QuantizedLinear) and hasattr(module, "layer_id"):
            fwd, bwd = _make_hooks(collector)
            handles.append(module.register_forward_hook(fwd))
            handles.append(module.register_full_backward_hook(bwd))
        elif (
            isinstance(module, SparseMoEBlock)
            and hasattr(module, "layer_id")
            and hasattr(module, "quantization_config")
        ):
            fwd, bwd = _make_moe_hooks(collector)
            handles.append(module.register_forward_hook(fwd))
            handles.append(module.register_full_backward_hook(bwd))
    return handles
