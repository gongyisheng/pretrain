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

# Largest normal exponent per fp8 element format; the shared power-of-two (MX-style)
# scale aligns a block's amax to this so the block's max value lands near the
# format's top.
_FP8_EMAX = {torch.float8_e4m3fn: 8, torch.float8_e5m2: 15}


def is_fp8(fmt: str) -> bool:
    return fmt in _FP8_DTYPE


def is_mxfp8(scaling: dict) -> bool:
    """Whether a resolved `scaling` dict selects the mxfp8 scheme (blockwise + E8M0)."""
    return (
        scaling.get("granularity") == "blockwise"
        and scaling.get("scale_dtype") == "fp8_e8m0"
    )


def is_int8s(fmt: str) -> bool:
    return fmt in _INT8S_QMAX


def is_quantized(fmt: str) -> bool:
    return is_fp8(fmt) or is_int8s(fmt)


def str_to_dtype_fp8(fmt: str) -> Optional[torch.dtype]:
    return _FP8_DTYPE.get(fmt)


def str_to_qmax_int8s(fmt: str) -> Optional[int]:
    return _INT8S_QMAX.get(fmt)


# --- element max map: qmax/emax/store dtype keyed by the quantized element `fmt`.
# These are properties of the element being quantized (not the scale), so `fmt`
# is the key; the scale scheme picks *which* of qmax/emax it aligns to.


def fmt_qmax(fmt: str) -> float:
    """Max representable magnitude of element `fmt` (fp8 finfo max, or int qmax)."""
    dt = _FP8_DTYPE.get(fmt)
    return float(torch.finfo(dt).max) if dt is not None else float(_INT8S_QMAX[fmt])


def fmt_emax(fmt: str) -> int:
    """Largest normal exponent of fp8 element `fmt`; 0 for int formats (unused there)."""
    dt = _FP8_DTYPE.get(fmt)
    return _FP8_EMAX[dt] if dt is not None else 0


def fmt_store_dtype(fmt: str) -> torch.dtype:
    """Storage dtype for element `fmt`: its fp8 dtype, or torch.int8 for int formats."""
    return _FP8_DTYPE.get(fmt, torch.int8)


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
