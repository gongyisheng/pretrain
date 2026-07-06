"""Quantization framework.

Config keywords the build actually implements live here (imported by
`QuantConfig` for validation). Only supported keywords are listed; roadmap
formats are noted as TODOs and added alongside their kernels.

Keep this module import-light (constants only): `src.utils.config` imports from
it, so importing heavy submodules (linear/utils) here would create a cycle. The
converter `apply_quantization` (Task 6) must therefore lazy-import its deps.
"""

from __future__ import annotations


QUANT_PASSTHROUGH = frozenset({"fp32", "fp16", "bf16"})

# Operand formats. fp8 variants quantize (`fp8` is an alias for `fp8_e4m3`).
# TODO: add "int4", "fp4_mx", "fp4_nv" as their kernels land.
QUANT_FORMATS = QUANT_PASSTHROUGH | frozenset({"fp8", "fp8_e4m3", "fp8_e5m2", "int8"})

# Shared scale granularities.
# TODO: add "blockwise" (with block_size) once its scaling kernels land.
QUANT_GRANULARITY = frozenset({"tensorwise", "rowwise"})

# Named dtype recipes: fill the per-operand `dtype` map with a standard setup.
# TODO: add "int4", "mxfp4", "nvfp4" recipes once supported.
QUANT_DTYPE_RECIPES = {
    "fp8": {
        "weight": "fp8_e4m3",
        "act": "fp8_e4m3",
        "input_grad": "fp8_e5m2",
        "weight_grad": "fp8_e5m2",
    },
    "int8": {
        "weight": "int8",
        "act": "int8",
        "input_grad": "int8",
        "weight_grad": "int8",
    },
}

QUANT_OPERANDS = ("weight", "act", "input_grad", "weight_grad")
