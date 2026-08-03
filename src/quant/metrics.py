from __future__ import annotations

import torch

from src.quant.constants import EPS
from src.quant.linear import QuantizedLinear


def compare_quantized(source_tensor, dequantized_tensor):
    """Quant health of an operand against its dequantized round-trip.

    `dequantized_tensor` is what the GEMM actually consumed, so nothing here
    re-derives a scale. Returns 0-dim float32 tensors on the source's device:
    sqnr (dB), flush_to_zero_rate, range_ratio. No host sync (no .item()).
    """
    source = source_tensor.float()
    dequantized = dequantized_tensor.float()
    magnitude = source.abs()
    error = (source - dequantized).norm().clamp_min(EPS)
    return {
        "sqnr": 20.0 * torch.log10(source.norm().clamp_min(EPS) / error),
        "flush_to_zero_rate": ((source != 0) & (dequantized == 0)).sum().float()
        / source.numel(),
        "range_ratio": magnitude.amax() / magnitude.median().clamp_min(EPS),
    }


class QuantMetricCollector:
    """Accumulates per-(module, site, metric) 0-dim tensors across a step's GEMMs.

    No host sync in `record`; `drain` performs the single `.tolist()` sync. Under
    gradient accumulation every micro-step writes the same keys, so a drained value
    is the last micro-batch's, not an average over the accumulation window.
    """

    def __init__(self):
        self.enabled = False
        self._store = {}  # (module_name, site, metric) -> 0-dim tensor

    def record(self, module_name, site, snapshot):
        metrics = compare_quantized(snapshot.source_tensor, snapshot.dequantize())
        for name, value in metrics.items():
            self._store[(module_name, site, name)] = value

    def drain(self):
        if not self._store:
            return {}
        keys = list(self._store.keys())
        values = torch.stack([self._store[k] for k in keys]).tolist()  # one sync
        self._store = {}
        return {
            f"quant/{metric}/{site}/{module_name}": value
            for (module_name, site, metric), value in zip(keys, values)
        }


class QuantProbe:
    """Recorder bound to one module's quantized GEMM sites.

    Keyed on the module's full name, so every projection reports separately —
    `blocks.3.mlp.down_proj` (whose activation is the post-activation hidden state)
    does not collide with `blocks.3.mlp.gate_up_proj`.
    """

    __slots__ = ("collector", "module_name")

    def __init__(self, collector, module_name):
        self.collector = collector
        self.module_name = module_name

    @property
    def enabled(self):
        return self.collector.enabled

    def record(self, site, snapshot):
        if snapshot is not None:  # None when that operand's format is passthrough
            self.collector.record(self.module_name, site, snapshot)


def attach_quant_probe(model, collector):
    """Give every quantized module a probe keyed on its full name in `model`."""
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            module.quantization_probe = QuantProbe(collector, name or "root")
