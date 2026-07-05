from __future__ import annotations

import warnings

import torch.nn as nn

from src.quant import QUANT_PASSTHROUGH, fp8
from src.quant.linear import QuantLinear
from src.quant.utils import resolve_rule

# Per-format hardware capability: fmt -> (predicate, requirement message). Formats
# absent here impose no requirement (e.g. passthrough dtypes, and int8 which runs
# on any modern GPU). Extend as int4/fp4 land.
# TODO: fold into a per-format backend when the backend registry is introduced.
_FMT_HARDWARE = {
    fmt: (fp8.is_supported, fp8.HARDWARE_REQUIREMENT)
    for fmt in ("fp8", "fp8_e4m3", "fp8_e5m2")
}


def _quantizes(rule) -> bool:
    """Whether a rule quantizes any operand (vs. all passthrough dtypes)."""
    return any(fmt not in QUANT_PASSTHROUGH for fmt in rule.dtype.values())


def _check_hardware(rules) -> None:
    """Raise if any enabled rule uses a fmt this hardware can't run."""
    for rule in rules:
        if not rule.enabled:
            continue
        for fmt in rule.dtype.values():
            requirement = _FMT_HARDWARE.get(fmt)
            if requirement is not None and not requirement[0]():
                raise RuntimeError(
                    f"quant dtype {fmt!r} requires {requirement[1]}. "
                    "Disable quant or run on supported hardware."
                )


def apply_quantization(model: nn.Module, config) -> nn.Module:
    """Swap eligible nn.Linear modules to QuantLinear per the run's quant rules.

    `config.training.quant` is a list of rules; each Linear takes the first
    enabled rule whose include/exclude matches its FQN (see `resolve_rule`).
    No-op when no rule is enabled. Must run BEFORE torch.compile so the tracer
    sees the swapped modules.
    """
    rules = config.training.quant
    if not any(rule.enabled for rule in rules):
        return model

    _check_hardware(rules)

    embedding_weight_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)
    }

    for parent_fqn, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            fqn = f"{parent_fqn}.{child_name}" if parent_fqn else child_name
            rule = resolve_rule(fqn, rules)
            if rule is None or not _quantizes(rule):
                continue
            if id(child.weight) in embedding_weight_ids:
                warnings.warn(
                    f"quant: skipping {fqn!r} — its weight is tied to an embedding; "
                    "swapping would break the tie."
                )
                continue
            setattr(parent, child_name, QuantLinear.from_linear(child, rule))
    return model
