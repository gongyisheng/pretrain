from __future__ import annotations

import fnmatch
from typing import Optional

import torch

from src.quant.constants import (
    QUANT_FORMATS,
    _FP8_FORMATS,
    _INT8_FORMATS,
    _STR_TO_DTYPE,
    _STR_TO_QMAX,
)
from src.utils.config import QuantizationConfig


def str_to_dtype(fmt: str) -> torch.dtype:
    return _STR_TO_DTYPE[fmt]


def str_to_qmax(fmt: str) -> float:
    return _STR_TO_QMAX[fmt]


def is_fp8(fmt: str) -> bool:
    return fmt in _FP8_FORMATS


def is_int8s(fmt: str) -> bool:
    return fmt in _INT8_FORMATS


def is_quantized(fmt: str) -> bool:
    return is_fp8(fmt) or is_int8s(fmt)


def is_supported(fmt: str) -> bool:
    if fmt not in QUANT_FORMATS:
        raise ValueError(
            f"not an element format: {fmt!r}; expected one of {sorted(QUANT_FORMATS)}"
        )
    if is_fp8(fmt):
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


def check_hardware_support(quantization_configs: list[QuantizationConfig]) -> None:
    for quantization_config in quantization_configs:
        if not quantization_config.enabled:
            continue
        for fmt in quantization_config.dtype.values():
            if not is_supported(fmt):
                raise RuntimeError(unsupported_error(fmt))


def should_quantize(fqn: str, cfg: QuantizationConfig) -> bool:

    def matches(patterns: list[str]) -> bool:
        leaf = fqn.rsplit(".", 1)[-1]
        return any(
            fnmatch.fnmatch(fqn, p) or fnmatch.fnmatch(leaf, p) for p in patterns
        )

    if cfg.include and not matches(cfg.include):
        return False
    return not matches(cfg.exclude)


def resolve_quantization_config(
    fqn: str, quantization_configs: list[QuantizationConfig]
) -> Optional[QuantizationConfig]:
    for quantization_config in quantization_configs:
        if quantization_config.enabled and should_quantize(fqn, quantization_config):
            return quantization_config
    return None
