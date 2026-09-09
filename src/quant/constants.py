import torch


# --- config vocabulary ------------------------------------------------------

QUANT_PASSTHROUGH = frozenset({"fp32", "fp16", "bf16"})

# "fp8" is a dtype recipe, not an element format: QUANT_DTYPE_RECIPES expands it
# to fp8_e4m3/fp8_e5m2 per operand, so it must never appear as an operand's dtype.
QUANT_FORMATS = QUANT_PASSTHROUGH | frozenset(
    {
        "fp8_e4m3",
        "fp8_e5m2",
        "fp4_e2m1",
        "fp4_e2m1_4over6",
        "int8",
        "int7",
        "int6",
        "int5",
        "int4",
    }
)

QUANT_GRANULARITY = frozenset({"tensorwise", "rowwise", "blockwise"})

# "RNE" is round-to-nearest-even.
# "SR" is stochastic rounding.
QUANT_ROUNDING = frozenset({"RNE", "SR"})

QUANT_SCALE_RECIPES = {
    "tensorwise": {"granularity": "tensorwise"},
    "rowwise": {"granularity": "rowwise"},
    "mxfp8": {
        "granularity": "blockwise",
        "block_shape": {
            "weight": (1, 32),
            "act": (1, 32),
            "grad_out": (1, 32),
        },
        "scale_dtype": "fp8_e8m0",
    },
    "nvfp4": {
        "granularity": "blockwise",
        "block_shape": {
            "weight": (16, 16),
            "act": (1, 16),
            "grad_out": (1, 16),
        },
        "scale_dtype": "fp8_e4m3",
        "enable_global_scale": True,
    },
}

QUANT_DTYPE_RECIPES = {
    "fp8": {"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e5m2"},
    "mxfp8": {"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "fp8_e4m3"},
    "nvfp4": {"weight": "fp4_e2m1", "act": "fp4_e2m1", "grad_out": "fp4_e2m1"},
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
    # E2M1 has no usable element dtype: two codes are packed into one byte.
    "fp4_e2m1": torch.uint8,
    "fp4_e2m1_4over6": torch.uint8,
    "int8": torch.int8,
    "int7": torch.int8,
    "int6": torch.int8,
    "int5": torch.int8,
    "int4": torch.int8,
}

_STR_TO_QMAX = {
    "fp8_e4m3": float(torch.finfo(torch.float8_e4m3fn).max),
    "fp8_e5m2": float(torch.finfo(torch.float8_e5m2).max),
    "fp4_e2m1": 6.0,
    "fp4_e2m1_4over6": 4.0,
    "int8": 127.0,
    "int7": 63.0,
    "int6": 31.0,
    "int5": 15.0,
    "int4": 7.0,
}

# The smallest positive magnitude each format represents: one code for an integer,
# the min subnormal for an fp8, matching that format's grid in _STR_TO_FP8_ULP. A
# narrow scale clamps up to this so it can never round to zero and divide its
# operand by zero.
_STR_TO_QMIN = {
    "fp8_e4m3": 2.0**-9,
    "fp8_e5m2": 2.0**-16,
    "fp4_e2m1": 0.5,
    "fp4_e2m1_4over6": 0.5,
    "int8": 1.0,
    "int7": 1.0,
    "int6": 1.0,
    "int5": 1.0,
    "int4": 1.0,
}

# (mantissa bits, log2 subnormal spacing) define each fp8 format's binade grid.
# New float formats must declare their grid explicitly.
_STR_TO_FP8_ULP = {
    "fp8_e4m3": (3, -9),
    "fp8_e5m2": (2, -16),
}

EPS = 1e-30

_FP8_FORMATS = frozenset({"fp8_e4m3", "fp8_e5m2"})
_FP4_FORMATS = frozenset({"fp4_e2m1", "fp4_e2m1_4over6"})
_INT8_FORMATS = frozenset({"int8", "int7", "int6", "int5", "int4"})

_FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
