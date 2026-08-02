from __future__ import annotations

import torch.nn as nn

from src.layers.mlp import SparseMoEBlock
from src.quant.constants import QUANT_PASSTHROUGH
from src.quant.linear import QuantizedLinear
from src.quant.moe import quantized_expert_mm
from src.quant.utils import resolve_quantization_config, check_hardware_support


def _is_passthrough(quantization_config) -> bool:
    return quantization_config is None or all(
        fmt in QUANT_PASSTHROUGH for fmt in quantization_config.dtype.values()
    )


def _layer_id_from_name(name: str) -> str:
    for part in name.split("."):
        if part.isdigit():
            return part
    return name


def apply_quantization(model: nn.Module, config) -> nn.Module:
    """Swap eligible nn.Linear modules to QuantizedLinear per the run's quant rules."""
    quantization_configs = config.training.quant
    if not any(qc.enabled for qc in quantization_configs):
        return model

    check_hardware_support(quantization_configs)

    embedding_weight_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)
    }

    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            quantization_config = resolve_quantization_config(
                full_name, quantization_configs
            )
            # skip configs that leave every operand in a passthrough dtype
            if _is_passthrough(quantization_config):
                continue
            if id(child.weight) in embedding_weight_ids:
                print(
                    f"quant: skipping {full_name!r} — its weight is tied to an "
                    "embedding; swapping would break the tie."
                )
                continue
            qmod = QuantizedLinear.from_linear(child, quantization_config)
            qmod.layer_id = _layer_id_from_name(full_name)
            setattr(parent, child_name, qmod)

    # MoE routed experts are stacked Parameters (not nn.Linear), so the swap loop
    # above can't reach them. Install a quantized expert_mm per matching rule.
    for name, module in model.named_modules():
        if not isinstance(module, SparseMoEBlock):
            continue
        quantization_config = resolve_quantization_config(name, quantization_configs)
        if _is_passthrough(quantization_config):
            continue
        module.expert_mm = quantized_expert_mm(quantization_config)
        module.layer_id = _layer_id_from_name(name)
        module.quantization_config = quantization_config
    return model
