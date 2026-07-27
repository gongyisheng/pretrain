from __future__ import annotations

import warnings

import torch.nn as nn

from src.quant.constants import QUANT_PASSTHROUGH
from src.quant.linear import QuantLinear
from src.quant.utils import resolve_rule, check_hardware_support


def apply_quantization(model: nn.Module, config) -> nn.Module:
    """Swap eligible nn.Linear modules to QuantLinear per the run's quant rules."""
    rules = config.training.quant
    if not any(rule.enabled for rule in rules):
        return model

    check_hardware_support(rules)

    embedding_weight_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)
    }

    for parent_fqn, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            fqn = f"{parent_fqn}.{child_name}" if parent_fqn else child_name
            rule = resolve_rule(fqn, rules)
            # skip rules that leave every operand in a passthrough dtype
            if rule is None or all(
                fmt in QUANT_PASSTHROUGH for fmt in rule.dtype.values()
            ):
                continue
            if id(child.weight) in embedding_weight_ids:
                warnings.warn(
                    f"quant: skipping {fqn!r} — its weight is tied to an embedding; "
                    "swapping would break the tie."
                )
                continue
            setattr(parent, child_name, QuantLinear.from_linear(child, rule))
    return model
