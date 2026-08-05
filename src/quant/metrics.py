from __future__ import annotations

import torch

from src.quant.quantize import dequantize_operand, ragged_scale_blocks
from src.utils.metric_utils import compute_quantization_metrics


class QuantizationMetricProbe:
    """The metrics one quantized module recorded this step, keyed by GEMM operand.
    Created and retained by a QuantizationMetricsCollector, which arms `enabled`
    and drains `metrics`; the probe holds no reference back to it.
    """

    __slots__ = ("enabled", "metrics")

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.metrics = {}  # gemm_operand -> {metric name -> 0-dim tensor}

    def record(self, gemm_operand, snapshot):
        if snapshot is None:  # that operand's format is passthrough
            return
        # A wgrad operand's scale rows tile each expert's rows, so reproduce the
        # mapping quantization used. Rebuilt rather than stored on the snapshot,
        # which holds only what the GEMM already had.
        ragged = None
        if snapshot.offs is not None and snapshot.contract_dim == 0:
            ragged = ragged_scale_blocks(
                snapshot.offs,
                snapshot.quantized_tensor.shape[0],
                snapshot.granularity,
                snapshot.block_size,
            )
        dequantized = dequantize_operand(
            snapshot.quantized_tensor,
            snapshot.scale,
            snapshot.contract_dim,
            snapshot.granularity,
            snapshot.block_size,
            ragged=ragged,
        )
        # offs makes the reduction per expert and averaged, so one hot expert cannot
        # set the number for a cold one
        self.metrics[gemm_operand] = compute_quantization_metrics(
            snapshot.source_tensor, dequantized, offs=snapshot.offs
        )


class QuantizationMetricsCollector:
    """Owns every probe, so `get_probe()` is the only place one is made and this is
    the only reference holding them alive.
    """

    def __init__(self):
        self._enabled = False
        self._probes = {}  # module_name -> QuantizationMetricProbe

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        for probe in self._probes.values():
            probe.enabled = value

    def get_probe(self, module_name):
        """The probe attributing `module_name`, created on first ask. Armed to the
        current gate, so a probe attached mid-run isn't silently dark for a step.
        """
        if module_name not in self._probes:
            self._probes[module_name] = QuantizationMetricProbe(self._enabled)
        return self._probes[module_name]

    def to_metrics_dict(self) -> dict[str, float]:
        keys, values = [], []
        for module_name, probe in self._probes.items():
            for gemm_operand, metrics in probe.metrics.items():
                for name, value in metrics.items():
                    keys.append((module_name, gemm_operand, name))
                    values.append(value)
            probe.metrics.clear()
        if not keys:
            return {}
        return {
            f"quant/{metric}/{gemm_operand}/{module_name}": value
            for (module_name, gemm_operand, metric), value in zip(
                keys,
                torch.stack(values).tolist(),  # one sync
            )
        }
