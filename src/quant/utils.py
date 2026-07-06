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


def is_fp8(fmt: str) -> bool:
    return fmt in _FP8_DTYPE


def is_int8(fmt: str) -> bool:
    return fmt == "int8"


def is_quantized(fmt: str) -> bool:
    return is_fp8(fmt) or is_int8(fmt)


def str_to_dtype_fp8(fmt: str) -> Optional[torch.dtype]:
    return _FP8_DTYPE.get(fmt)


def should_quantize(fqn: str, cfg: QuantConfig) -> bool:

    def matches(patterns: list[str]) -> bool:
        leaf = fqn.rsplit(".", 1)[-1]
        return any(
            fnmatch.fnmatch(fqn, p) or fnmatch.fnmatch(leaf, p) for p in patterns
        )

    if cfg.include and not matches(cfg.include):
        return False
    return not matches(cfg.exclude)


def resolve_rule(fqn: str, rules: list[QuantConfig]) -> Optional[QuantConfig]:

    for rule in rules:
        if rule.enabled and should_quantize(fqn, rule):
            return rule
    return None
