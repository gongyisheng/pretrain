from __future__ import annotations

import fnmatch
from typing import Optional

import torch

from src.quant.constants import (
    _FP8_FORMATS,
    _INT8_FORMATS,
    _STR_TO_DTYPE,
    _STR_TO_EMAX,
    _STR_TO_QMAX,
)
from src.utils.config import QuantConfig


def str_to_dtype(fmt: str) -> Optional[torch.dtype]:
    return _STR_TO_DTYPE.get(fmt)


def str_to_store_dtype(fmt: str) -> torch.dtype:
    return _STR_TO_DTYPE.get(fmt) or torch.int8


def str_to_qmax(fmt: str) -> float:
    return _STR_TO_QMAX[fmt]


def str_to_emax(fmt: str) -> int:
    return _STR_TO_EMAX[fmt]


def is_fp8(fmt: str) -> bool:
    return fmt in _FP8_FORMATS


def is_int8s(fmt: str) -> bool:
    return fmt in _INT8_FORMATS


def is_quantized(fmt: str) -> bool:
    return is_fp8(fmt) or is_int8s(fmt)


def is_supported(fmt: str) -> bool:
    if str_to_dtype(fmt) is not None:
        return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
            8,
            9,
        )
    return True


def unsupported_error(fmt: str) -> str:
    requirement = {
        "fp8": "a CUDA GPU with compute capability >= 8.9 (Ada/Hopper/Blackwell)",
        # TODO: "fp4": "a CUDA GPU with compute capability >= 10.0 (Blackwell)",
    }.get(fmt.split("_", 1)[0], "supported hardware")
    return (
        f"quant dtype {fmt!r} requires {requirement}. "
        "Disable quant or run on supported hardware."
    )


def check_hardware_support(rules: list[QuantConfig]) -> None:
    for rule in rules:
        if not rule.enabled:
            continue
        for fmt in rule.dtype.values():
            if not is_supported(fmt):
                raise RuntimeError(unsupported_error(fmt))


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
