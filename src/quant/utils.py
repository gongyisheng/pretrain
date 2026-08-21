from __future__ import annotations

import fnmatch
from typing import Optional

import torch

from src.quant.constants import (
    _FP8_FORMATS,
    _INT8_FORMATS,
    _STR_TO_DTYPE,
    _STR_TO_FP8_ULP,
    _STR_TO_QMAX,
)
from src.utils.config import QuantizationConfig


def str_to_dtype(fmt: str) -> torch.dtype:
    return _STR_TO_DTYPE[fmt]


def str_to_qmax(fmt: str) -> float:
    return _STR_TO_QMAX[fmt]


def str_to_fp8_ulp(fmt: str) -> tuple[int, int]:
    """(mantissa bits, log2 of the subnormal spacing) for an fp8 format."""
    return _STR_TO_FP8_ULP[fmt]


def is_fp8(fmt: str) -> bool:
    return fmt in _FP8_FORMATS


def is_int8s(fmt: str) -> bool:
    return fmt in _INT8_FORMATS


def is_quantized(fmt: str) -> bool:
    return is_fp8(fmt) or is_int8s(fmt)


def should_quantize(fqn: str, cfg: QuantizationConfig) -> bool:

    def matches(patterns: list[str]) -> bool:
        leaf = fqn.rsplit(".", 1)[-1]
        return any(
            fnmatch.fnmatch(fqn, p) or fnmatch.fnmatch(leaf, p) for p in patterns
        )

    if cfg.include and not matches(cfg.include):
        return False
    return not matches(cfg.exclude)


def _resolve_gemm_quantization_family(
    a_fmt: str,
    b_fmt: str,
    scale_dtype: torch.dtype,
    block_shape: tuple[int, int],
) -> str | None:
    if is_fp8(a_fmt) and is_fp8(b_fmt):
        if scale_dtype == torch.float8_e8m0fnu:
            return "mxfp8"
        return "fp8"
    if is_int8s(a_fmt) and is_int8s(b_fmt):
        return "int8"
    return None


def scaled_mm_op(
    a_fmt: str,
    b_fmt: str,
    scale_dtype: torch.dtype,
    block_shape: tuple[int, int],
) -> str | None:
    family = _resolve_gemm_quantization_family(a_fmt, b_fmt, scale_dtype, block_shape)
    return None if family is None else f"gemm.{family}_scaled_mm"


def scaled_grouped_mm_op(
    a_fmt: str,
    b_fmt: str,
    scale_dtype: torch.dtype,
    block_shape: tuple[int, int],
) -> str | None:
    family = _resolve_gemm_quantization_family(a_fmt, b_fmt, scale_dtype, block_shape)
    return None if family is None else f"gemm.{family}_scaled_grouped_mm"


def resolve_quantization_config(
    fqn: str, quantization_configs: list[QuantizationConfig]
) -> Optional[QuantizationConfig]:
    for quantization_config in quantization_configs:
        if quantization_config.enabled and should_quantize(fqn, quantization_config):
            return quantization_config
    return None
