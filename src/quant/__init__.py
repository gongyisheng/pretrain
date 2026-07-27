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
# The int8 series (int8/int7/int6/int5/int4) all store in torch.int8 and compute
# via torch._int_mm; the format only sets the value range (qmax = 127/63/31/15/7).
# TODO: add fp4_mx, fp4_nv.
QUANT_FORMATS = QUANT_PASSTHROUGH | frozenset(
    {"fp8", "fp8_e4m3", "fp8_e5m2", "int8", "int7", "int6", "int5", "int4"}
)

# Shared scale granularities. "blockwise" needs `block_size` (+ `scale_dtype`
# for MX); see QUANT_SCALING_RECIPES.
QUANT_GRANULARITY = frozenset({"tensorwise", "rowwise", "blockwise"})

# Named scale schemes: fill the `scaling` dict with a standard granularity /
# block / scale-dtype setup, independent of the element format. "mxfp8" = OCP
# Microscaling FP8 (block of 32 along K, power-of-two E8M0 scale) — the block/
# scale half of the mxfp8 recipe, paired with an fp8 element.
# TODO: add "mxfp4" (block 32) / "nvfp4" (block 16, e4m3 scale) recipes.
QUANT_SCALING_RECIPES = {
    "tensorwise": {"granularity": "tensorwise"},
    "rowwise": {"granularity": "rowwise"},
    "blockwise": {"granularity": "blockwise", "block_size": 128, "scale_dtype": "fp32"},
    "mxfp8": {"granularity": "blockwise", "block_size": 32, "scale_dtype": "fp8_e8m0"},
}

# Named dtype recipes: fill the per-operand `dtype` map with a standard setup.
# The "mxfp8" dtype recipe also seeds the matching "mxfp8" scale scheme (see
# QuantConfig.__post_init__). Unlike plain fp8 (e5m2 grads), mxfp8 uses e4m3 for
# every operand: the mxfp8 BlockWise1x32 GEMM only accepts e4m3 x e4m3, and its
# per-32-block rescaling
# gives e4m3 enough dynamic range for gradients — so all three training GEMMs
# stay on the Blackwell tensor-core path.
QUANT_DTYPE_RECIPES = {
    "fp8": {
        "weight": "fp8_e4m3",
        "act": "fp8_e4m3",
        "input_grad": "fp8_e5m2",
        "weight_grad": "fp8_e5m2",
    },
    "mxfp8": {
        "weight": "fp8_e4m3",
        "act": "fp8_e4m3",
        "input_grad": "fp8_e4m3",
        "weight_grad": "fp8_e4m3",
    },
    **{
        fmt: {op: fmt for op in ("weight", "act", "input_grad", "weight_grad")}
        for fmt in ("int8", "int7", "int6", "int5", "int4")
    },
}

QUANT_OPERANDS = ("weight", "act", "input_grad", "weight_grad")
