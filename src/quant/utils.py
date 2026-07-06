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

# Symmetric max magnitude per int fmt = 2^(bits-1) - 1. All stored in torch.int8.
_INT8S_QMAX = {"int8": 127, "int7": 63, "int6": 31, "int5": 15, "int4": 7}


def is_fp8(fmt: str) -> bool:
    return fmt in _FP8_DTYPE


def is_int8s(fmt: str) -> bool:
    return fmt in _INT8S_QMAX


def is_quantized(fmt: str) -> bool:
    return is_fp8(fmt) or is_int8s(fmt)


def str_to_dtype_fp8(fmt: str) -> Optional[torch.dtype]:
    return _FP8_DTYPE.get(fmt)


def str_to_qmax_int8s(fmt: str) -> Optional[int]:
    return _INT8S_QMAX.get(fmt)


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
