from __future__ import annotations

import fnmatch
from typing import Optional

import torch

from src.utils.config import QuantConfig


_FP8_DTYPE = {
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def quant_dtype(fmt: str) -> Optional[torch.dtype]:
    """The fp8 dtype an operand with this fmt quantizes to, or None when the
    fmt is a passthrough dtype (fp32/fp16/bf16) that stays high precision."""
    return _FP8_DTYPE.get(fmt)


def should_quantize(fqn: str, cfg: QuantConfig) -> bool:
    """Whether the rule `cfg` applies to the module at `fqn`.

    Matches iff (include empty OR fqn matches an include glob) AND fqn matches
    no exclude glob. Exclude wins over include. Globs match either the full FQN
    or the leaf name, so bare patterns like "q_proj" hit any depth.
    """

    def matches(patterns: list[str]) -> bool:
        leaf = fqn.rsplit(".", 1)[-1]
        return any(
            fnmatch.fnmatch(fqn, p) or fnmatch.fnmatch(leaf, p) for p in patterns
        )

    if cfg.include and not matches(cfg.include):
        return False
    return not matches(cfg.exclude)


def resolve_rule(fqn: str, rules: list[QuantConfig]) -> Optional[QuantConfig]:
    """First enabled rule that applies to `fqn`, or None (leave high precision).

    Rules are tried in order, so put more specific rules first."""
    for rule in rules:
        if rule.enabled and should_quantize(fqn, rule):
            return rule
    return None
