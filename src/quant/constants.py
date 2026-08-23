from __future__ import annotations

import torch


# --- config vocabulary ------------------------------------------------------

QUANT_PASSTHROUGH = frozenset({"fp32", "fp16", "bf16"})

# "fp8" is a dtype recipe, not an element format: QUANT_DTYPE_RECIPES expands it
# to fp8_e4m3/fp8_e5m2 per operand, so it must never appear as an operand's dtype.
QUANT_FORMATS = QUANT_PASSTHROUGH | frozenset(
    {"fp8_e4m3", "fp8_e5m2", "int8", "int7", "int6", "int5", "int4"}
)

QUANT_GRANULARITY = frozenset({"tensorwise", "rowwise", "blockwise"})

# "RNE" is round-to-nearest-even.
# "SR" is stochastic rounding.
QUANT_ROUNDING = frozenset({"RNE", "SR"})

QUANT_SCALE_RECIPES = {
    "tensorwise": {"granularity": "tensorwise"},
    "rowwise": {"granularity": "rowwise"},
    "blockwise": {
        "granularity": "blockwise",
        "block_shape": (1, 128),
        "scale_dtype": "fp32",
    },
    "mxfp8": {
        "granularity": "blockwise",
        "block_shape": (1, 32),
        "scale_dtype": "fp8_e8m0",
    },
}

QUANT_DTYPE_RECIPES = {
    "fp8": {"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
    "mxfp8": {"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e4m3"},
    **{
        fmt: {"weight": fmt, "act": "bf16", "grad_out": "bf16"}
        for fmt in ("int8", "int7", "int6", "int5", "int4")
    },
}

# The three GEMM operations and tensors held by a differentiable linear.
GEMM_OPS = ("fwd", "dgrad", "wgrad")
GEMM_TENSORS = ("weight", "act", "grad_out")

# Each tensor is consumed by two operations. A dtype is a property of a tensor;
# scoping it to an operation is how one tensor takes two dtypes:
#
#   fwd    y  = act @ weightᵀ
#   dgrad  dx = grad_out @ weight
#   wgrad  dw = grad_outᵀ @ act
GEMM_OPS_BY_TENSOR = {
    "weight": ("fwd", "dgrad"),
    "act": ("fwd", "wgrad"),
    "grad_out": ("dgrad", "wgrad"),
}

# --- format property maps ---------------------------------------------------

_STR_TO_DTYPE = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e8m0": torch.float8_e8m0fnu,
    "int8": torch.int8,
    "int7": torch.int8,
    "int6": torch.int8,
    "int5": torch.int8,
    "int4": torch.int8,
}

_STR_TO_QMAX = {
    "fp8_e4m3": float(torch.finfo(torch.float8_e4m3fn).max),
    "fp8_e5m2": float(torch.finfo(torch.float8_e5m2).max),
    "int8": 127.0,
    "int7": 63.0,
    "int6": 31.0,
    "int5": 15.0,
    "int4": 7.0,
}

# (mantissa bits, log2 subnormal spacing) define each fp8 format's binade grid.
# New float formats must declare their grid explicitly.
_STR_TO_FP8_ULP = {
    "fp8_e4m3": (3, -9),
    "fp8_e5m2": (2, -16),
}

EPS = 1e-30

# Operand formats only: fp8_e8m0 is a scale dtype, never an operand's.
_FP8_FORMATS = frozenset({"fp8_e4m3", "fp8_e5m2"})
_INT8_FORMATS = frozenset({"int8", "int7", "int6", "int5", "int4"})
