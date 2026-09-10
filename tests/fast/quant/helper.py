"""Quantization test helpers."""

from itertools import product

import pytest
import torch

from src.quant.constants import (
    QUANT_PASSTHROUGH,
    _FP4_FORMATS,
    _FP8_FORMATS,
    _INT8_FORMATS,
)
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.utils import is_fp4, is_quantized, scaled_mm_op
from src.utils.config import TrainingConfig

E4M3 = "fp8_e4m3"
FP8_FORMATS = sorted(_FP8_FORMATS)
FP4_FORMATS = sorted(_FP4_FORMATS)
INT_FORMATS = sorted(_INT8_FORMATS)
ALL_QUANT_FORMATS = FP8_FORMATS + INT_FORMATS + FP4_FORMATS
PASSTHROUGH_FORMATS = sorted(QUANT_PASSTHROUGH)
ALL_FORMATS = ALL_QUANT_FORMATS + PASSTHROUGH_FORMATS
FP8_E4M3_W8A8_DTYPES = {"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "bf16"}
FP8_E4M3_W16A8_DTYPES = {"weight": "bf16", "act": "fp8_e4m3", "grad_out": "bf16"}
FP8_E4M3_W8A16_DTYPES = {"weight": "fp8_e4m3", "act": "bf16", "grad_out": "bf16"}
FP8_E5M2_W8A8_DTYPES = {"weight": "fp8_e5m2", "act": "fp8_e5m2", "grad_out": "bf16"}
FP8_E5M2_W16A8_DTYPES = {"weight": "bf16", "act": "fp8_e5m2", "grad_out": "bf16"}
FP8_E5M2_W8A16_DTYPES = {"weight": "fp8_e5m2", "act": "bf16", "grad_out": "bf16"}
FP8_E4M3_W8A8G8_DTYPES = {
    "weight": "fp8_e4m3",
    "act": "fp8_e4m3",
    "grad_out": "fp8_e4m3",
}
FP8_E4M3_W8A8_E5M2_G8_DTYPES = {
    "weight": "fp8_e4m3",
    "act": "fp8_e4m3",
    "grad_out": "fp8_e5m2",
}
FP8_E5M2_W8A8G8_DTYPES = {
    "weight": "fp8_e5m2",
    "act": "fp8_e5m2",
    "grad_out": "fp8_e5m2",
}
FP8_E5M2_W8A8_E4M3_G8_DTYPES = {
    "weight": "fp8_e5m2",
    "act": "fp8_e5m2",
    "grad_out": "fp8_e4m3",
}
INT8_W8A8_DTYPES = {"weight": "int8", "act": "int8", "grad_out": "bf16"}
INT8_W8A8G8_DTYPES = {"weight": "int8", "act": "int8", "grad_out": "int8"}
INT8_W16A8_DTYPES = {"weight": "bf16", "act": "int8", "grad_out": "bf16"}
INT8_W8A16_DTYPES = {"weight": "int8", "act": "bf16", "grad_out": "bf16"}
INT7_W8A16_DTYPES = {"weight": "int7", "act": "bf16", "grad_out": "bf16"}
INT6_W8A16_DTYPES = {"weight": "int6", "act": "bf16", "grad_out": "bf16"}
INT5_W8A16_DTYPES = {"weight": "int5", "act": "bf16", "grad_out": "bf16"}
INT4_W8A16_DTYPES = {"weight": "int4", "act": "bf16", "grad_out": "bf16"}
FP4_E2M1_W4A4G4_DTYPES = {
    "weight": "fp4_e2m1",
    "act": "fp4_e2m1",
    "grad_out": "fp4_e2m1",
}
FP4_E2M1_W4A4_DTYPES = {"weight": "fp4_e2m1", "act": "fp4_e2m1", "grad_out": "bf16"}
FP4_E2M1_W4A16_DTYPES = {"weight": "fp4_e2m1", "act": "bf16", "grad_out": "bf16"}
FP4_E2M1_4OVER6_W4A4G4_DTYPES = {
    "weight": "fp4_e2m1_4over6",
    "act": "fp4_e2m1_4over6",
    "grad_out": "fp4_e2m1_4over6",
}
FP4_E2M1_4OVER6_W4A4_DTYPES = {
    "weight": "fp4_e2m1_4over6",
    "act": "fp4_e2m1_4over6",
    "grad_out": "bf16",
}
FP4_E2M1_4OVER6_W4A16_DTYPES = {
    "weight": "fp4_e2m1_4over6",
    "act": "bf16",
    "grad_out": "bf16",
}

FP8_E4M3_W8A8G8_GWHP_DTYPES = {
    "weight": "fp8_e4m3",
    "act": {"fwd": "fp8_e4m3", "wgrad": "bf16"},
    "grad_out": {"dgrad": "fp8_e4m3", "wgrad": "bf16"},
}
FP8_E4M3_W8A8G8_GIHP_DTYPES = {
    "weight": {"fwd": "fp8_e4m3", "dgrad": "bf16"},
    "act": "fp8_e4m3",
    "grad_out": {"dgrad": "bf16", "wgrad": "fp8_e4m3"},
}
FP8_E4M3_W8A8_E5M2_G8_GWHP_DTYPES = {
    "weight": "fp8_e4m3",
    "act": {"fwd": "fp8_e4m3", "wgrad": "bf16"},
    "grad_out": {"dgrad": "fp8_e5m2", "wgrad": "bf16"},
}
FP8_E4M3_W8A8_E5M2_G8_GIHP_DTYPES = {
    "weight": {"fwd": "fp8_e4m3", "dgrad": "bf16"},
    "act": "fp8_e4m3",
    "grad_out": {"dgrad": "bf16", "wgrad": "fp8_e5m2"},
}
FP8_E5M2_W8A8G8_GWHP_DTYPES = {
    "weight": "fp8_e5m2",
    "act": {"fwd": "fp8_e5m2", "wgrad": "bf16"},
    "grad_out": {"dgrad": "fp8_e5m2", "wgrad": "bf16"},
}
FP8_E5M2_W8A8G8_GIHP_DTYPES = {
    "weight": {"fwd": "fp8_e5m2", "dgrad": "bf16"},
    "act": "fp8_e5m2",
    "grad_out": {"dgrad": "bf16", "wgrad": "fp8_e5m2"},
}
FP8_E5M2_W8A8_E4M3_G8_GWHP_DTYPES = {
    "weight": "fp8_e5m2",
    "act": {"fwd": "fp8_e5m2", "wgrad": "bf16"},
    "grad_out": {"dgrad": "fp8_e4m3", "wgrad": "bf16"},
}
FP8_E5M2_W8A8_E4M3_G8_GIHP_DTYPES = {
    "weight": {"fwd": "fp8_e5m2", "dgrad": "bf16"},
    "act": "fp8_e5m2",
    "grad_out": {"dgrad": "bf16", "wgrad": "fp8_e4m3"},
}
INT8_W8A8G8_GWHP_DTYPES = {
    "weight": "int8",
    "act": {"fwd": "int8", "wgrad": "bf16"},
    "grad_out": {"dgrad": "int8", "wgrad": "bf16"},
}
INT8_W8A8G8_GIHP_DTYPES = {
    "weight": {"fwd": "int8", "dgrad": "bf16"},
    "act": "int8",
    "grad_out": {"dgrad": "bf16", "wgrad": "int8"},
}


FORWARD_DTYPES = [
    FP4_E2M1_W4A4_DTYPES,
    FP4_E2M1_W4A16_DTYPES,
    FP4_E2M1_4OVER6_W4A4_DTYPES,
    FP4_E2M1_4OVER6_W4A16_DTYPES,
    FP8_E4M3_W8A8_DTYPES,
    FP8_E4M3_W16A8_DTYPES,
    FP8_E4M3_W8A16_DTYPES,
    INT8_W8A8_DTYPES,
    INT8_W16A8_DTYPES,
    INT8_W8A16_DTYPES,
    INT7_W8A16_DTYPES,
    INT6_W8A16_DTYPES,
    INT5_W8A16_DTYPES,
    INT4_W8A16_DTYPES,
]

BACKWARD_DTYPES = [
    FP4_E2M1_W4A4_DTYPES,
    FP4_E2M1_W4A4G4_DTYPES,
    FP4_E2M1_W4A16_DTYPES,
    FP4_E2M1_4OVER6_W4A4_DTYPES,
    FP4_E2M1_4OVER6_W4A4G4_DTYPES,
    FP4_E2M1_4OVER6_W4A16_DTYPES,
    FP8_E4M3_W8A8_DTYPES,
    FP8_E4M3_W16A8_DTYPES,
    FP8_E4M3_W8A16_DTYPES,
    FP8_E5M2_W8A8_DTYPES,
    FP8_E5M2_W16A8_DTYPES,
    FP8_E5M2_W8A16_DTYPES,
    FP8_E4M3_W8A8G8_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_DTYPES,
    FP8_E5M2_W8A8G8_DTYPES,
    FP8_E5M2_W8A8_E4M3_G8_DTYPES,
    INT8_W8A8_DTYPES,
    INT8_W16A8_DTYPES,
    INT8_W8A16_DTYPES,
    INT7_W8A16_DTYPES,
    INT6_W8A16_DTYPES,
    INT5_W8A16_DTYPES,
    INT4_W8A16_DTYPES,
    FP8_E4M3_W8A8G8_GWHP_DTYPES,
    FP8_E4M3_W8A8G8_GIHP_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_GWHP_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_GIHP_DTYPES,
    FP8_E5M2_W8A8G8_GWHP_DTYPES,
    FP8_E5M2_W8A8G8_GIHP_DTYPES,
    FP8_E5M2_W8A8_E4M3_G8_GWHP_DTYPES,
    FP8_E5M2_W8A8_E4M3_G8_GIHP_DTYPES,
    INT8_W8A8G8_GWHP_DTYPES,
    INT8_W8A8G8_GIHP_DTYPES,
]


def scale_of(
    granularity,
    block_shape=(0, 0),
    scale_dtype=torch.float32,
    enable_global_scale=False,
):
    return {
        "granularity": granularity,
        "block_shape": block_shape,
        "scale_dtype": scale_dtype,
        "enable_global_scale": enable_global_scale,
    }


TENSORWISE = scale_of("tensorwise")
ROWWISE = scale_of("rowwise")
# Blockwise: contract extents are multiples of 16; outer extents are 1 or square.
BLOCKWISE1D_16 = scale_of("blockwise", (1, 16))
BLOCKWISE1D_32 = scale_of("blockwise", (1, 32))
BLOCKWISE1D_64 = scale_of("blockwise", (1, 64))
BLOCKWISE1D_128 = scale_of("blockwise", (1, 128))
BLOCKWISE2D_16 = scale_of("blockwise", (16, 16))
BLOCKWISE2D_32 = scale_of("blockwise", (32, 32))
BLOCKWISE2D_64 = scale_of("blockwise", (64, 64))
BLOCKWISE2D_128 = scale_of("blockwise", (128, 128))
BLOCKWISE1D_16_E8M0 = scale_of("blockwise", (1, 16), torch.float8_e8m0fnu)
BLOCKWISE1D_32_E8M0 = scale_of("blockwise", (1, 32), torch.float8_e8m0fnu)
BLOCKWISE1D_64_E8M0 = scale_of("blockwise", (1, 64), torch.float8_e8m0fnu)
BLOCKWISE1D_128_E8M0 = scale_of("blockwise", (1, 128), torch.float8_e8m0fnu)
BLOCKWISE2D_16_E8M0 = scale_of("blockwise", (16, 16), torch.float8_e8m0fnu)
BLOCKWISE2D_32_E8M0 = scale_of("blockwise", (32, 32), torch.float8_e8m0fnu)
BLOCKWISE2D_64_E8M0 = scale_of("blockwise", (64, 64), torch.float8_e8m0fnu)
BLOCKWISE2D_128_E8M0 = scale_of("blockwise", (128, 128), torch.float8_e8m0fnu)
# An e4m3 scale is defined over every element format, unlike e8m0.
ROWWISE_E4M3 = scale_of("rowwise", scale_dtype=torch.float8_e4m3fn)
BLOCKWISE1D_16_E4M3 = scale_of("blockwise", (1, 16), torch.float8_e4m3fn)
BLOCKWISE1D_32_E4M3 = scale_of("blockwise", (1, 32), torch.float8_e4m3fn)
BLOCKWISE1D_64_E4M3 = scale_of("blockwise", (1, 64), torch.float8_e4m3fn)
BLOCKWISE1D_128_E4M3 = scale_of("blockwise", (1, 128), torch.float8_e4m3fn)
BLOCKWISE2D_16_E4M3 = scale_of("blockwise", (16, 16), torch.float8_e4m3fn)
BLOCKWISE2D_32_E4M3 = scale_of("blockwise", (32, 32), torch.float8_e4m3fn)
BLOCKWISE2D_64_E4M3 = scale_of("blockwise", (64, 64), torch.float8_e4m3fn)
BLOCKWISE2D_128_E4M3 = scale_of("blockwise", (128, 128), torch.float8_e4m3fn)
# E2M1 uses e4m3 block scales under a per-tensor fp32 global scale.
BLOCKWISE1D_16_E2M1 = scale_of(
    "blockwise", (1, 16), torch.float8_e4m3fn, enable_global_scale=True
)
BLOCKWISE1D_32_E2M1 = scale_of(
    "blockwise", (1, 32), torch.float8_e4m3fn, enable_global_scale=True
)
BLOCKWISE1D_64_E2M1 = scale_of(
    "blockwise", (1, 64), torch.float8_e4m3fn, enable_global_scale=True
)
BLOCKWISE1D_128_E2M1 = scale_of(
    "blockwise", (1, 128), torch.float8_e4m3fn, enable_global_scale=True
)
BLOCKWISE2D_16_E2M1 = scale_of(
    "blockwise", (16, 16), torch.float8_e4m3fn, enable_global_scale=True
)
BLOCKWISE2D_32_E2M1 = scale_of(
    "blockwise", (32, 32), torch.float8_e4m3fn, enable_global_scale=True
)
BLOCKWISE2D_64_E2M1 = scale_of(
    "blockwise", (64, 64), torch.float8_e4m3fn, enable_global_scale=True
)
BLOCKWISE2D_128_E2M1 = scale_of(
    "blockwise", (128, 128), torch.float8_e4m3fn, enable_global_scale=True
)
ROWWISE_E2M1 = scale_of(
    "rowwise", scale_dtype=torch.float8_e4m3fn, enable_global_scale=True
)


ALL_SCALES = [
    TENSORWISE,
    ROWWISE,
    BLOCKWISE1D_16,
    BLOCKWISE1D_32,
    BLOCKWISE1D_64,
    BLOCKWISE1D_128,
    BLOCKWISE2D_16,
    BLOCKWISE2D_32,
    BLOCKWISE2D_64,
    BLOCKWISE2D_128,
    BLOCKWISE1D_16_E8M0,
    BLOCKWISE1D_32_E8M0,
    BLOCKWISE1D_64_E8M0,
    BLOCKWISE1D_128_E8M0,
    BLOCKWISE2D_16_E8M0,
    BLOCKWISE2D_32_E8M0,
    BLOCKWISE2D_64_E8M0,
    BLOCKWISE2D_128_E8M0,
    ROWWISE_E4M3,
    BLOCKWISE1D_16_E4M3,
    BLOCKWISE1D_32_E4M3,
    BLOCKWISE1D_64_E4M3,
    BLOCKWISE1D_128_E4M3,
    BLOCKWISE2D_16_E4M3,
    BLOCKWISE2D_32_E4M3,
    BLOCKWISE2D_64_E4M3,
    BLOCKWISE2D_128_E4M3,
    BLOCKWISE1D_16_E2M1,
    BLOCKWISE1D_32_E2M1,
    BLOCKWISE1D_64_E2M1,
    BLOCKWISE1D_128_E2M1,
    BLOCKWISE2D_16_E2M1,
    BLOCKWISE2D_32_E2M1,
    BLOCKWISE2D_64_E2M1,
    BLOCKWISE2D_128_E2M1,
    ROWWISE_E2M1,
]

# Explicit operand layouts for the GEMM scale sweeps.
BASE_SCALES = [
    TENSORWISE,
    ROWWISE,
    BLOCKWISE1D_16,
    BLOCKWISE1D_128,
    BLOCKWISE2D_16,
    BLOCKWISE2D_128,
    BLOCKWISE1D_16_E8M0,
    BLOCKWISE1D_128_E8M0,
    BLOCKWISE2D_16_E8M0,
    BLOCKWISE2D_128_E8M0,
    ROWWISE_E4M3,
    BLOCKWISE1D_16_E4M3,
    BLOCKWISE1D_128_E4M3,
    BLOCKWISE2D_16_E4M3,
    BLOCKWISE2D_128_E4M3,
    BLOCKWISE1D_16_E2M1,
    BLOCKWISE1D_128_E2M1,
    BLOCKWISE2D_16_E2M1,
    BLOCKWISE2D_128_E2M1,
    ROWWISE_E2M1,
]


def scale_combinations(scales, n_operands):
    return [
        operands
        for operands in product(scales, repeat=n_operands)
        if len(
            {
                (
                    scale["granularity"],
                    scale["block_shape"][1],
                    scale["scale_dtype"],
                    scale["enable_global_scale"],
                )
                for scale in operands
            }
        )
        == 1
    ]


# Only outer extents vary: GEMMs share scale dtype, global scaling, and K width.
SCALE_PAIRS = scale_combinations(BASE_SCALES, 2)
SCALE_TRIPLES = scale_combinations(BASE_SCALES, 3)


SCALES_COARSE_TO_FINE = [
    TENSORWISE,
    ROWWISE,
    BLOCKWISE1D_128,
    BLOCKWISE1D_64,
    BLOCKWISE1D_32,
    BLOCKWISE1D_16,
]

SCALE_DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float8_e8m0fnu: "fp8_e8m0",
    torch.float8_e4m3fn: "fp8_e4m3",
}


def roundtrip(x, contract_dim, fmt, scale_cfg, rotation=None):
    """Return a quantized operand after dequantization."""
    if not is_quantized(fmt):
        return x
    xq, scale, g = quantize_operand(x, contract_dim, fmt, scale_cfg, rotation=rotation)
    return dequantize_operand(
        xq, scale, contract_dim, scale_cfg, rotation=rotation, global_scale=g
    )


def fused_op_exists(a_fmt, b_fmt, scale_cfg):
    """Whether both formats use the same fused scaled-GEMM family."""
    return (
        scaled_mm_op(
            a_fmt,
            b_fmt,
            scale_cfg["scale_dtype"],
            scale_cfg["block_shape"],
        )
        is not None
    )


def uses_fp4_gemm(a_fmt, b_fmt, scale_cfg):
    return is_fp4(a_fmt) and is_fp4(b_fmt) and fused_op_exists(a_fmt, b_fmt, scale_cfg)


def mm_ref(a, b, a_fmt, b_fmt, a_scale, b_scale, rotation=None):
    """Reference GEMM with fused and fallback precision behavior.

    Fused GEMMs accumulate in fp32. The fallback first restores operand dtype, then
    accumulates in fp32. Return fp32 so callers add bias and cast only once.
    """
    dtype = torch.float32 if fused_op_exists(a_fmt, b_fmt, a_scale) else a.dtype
    return (
        roundtrip(a, -1, a_fmt, a_scale, rotation=rotation).to(dtype).float()
        @ roundtrip(b, -2, b_fmt, b_scale, rotation=rotation).to(dtype).float()
    )


def skip_unsupported_fmt_scale(fmt, scale_cfg):
    """Skip unsupported E8M0 element formats and block widths."""
    if (
        scale_cfg["scale_dtype"] is torch.float8_e8m0fnu
        and scale_cfg["block_shape"][1] % 32
    ):
        pytest.skip("E8M0 scales require a block width divisible by 32")
    if (
        scale_cfg["scale_dtype"] is torch.float8_e8m0fnu
        and is_quantized(fmt)
        and fmt not in FP8_FORMATS
    ):
        pytest.skip("an e8m0 scale is defined only over fp8 elements")


def skip_unsupported_ragged_k_scale(scale_cfg):
    """Skip a wider-than-32 E8M0 block on a ragged K axis."""
    if (
        scale_cfg["scale_dtype"] is torch.float8_e8m0fnu
        and scale_cfg["block_shape"][1] != 32
    ):
        pytest.skip("mxfp8 ragged-K needs a 32-wide scale block")


def skip_unsupported_dtype_scale(dtype, scale_cfg):
    """Skip format/scale combinations that QuantizationConfig rejects."""
    for spec in dtype.values():
        for fmt in spec.values() if isinstance(spec, dict) else (spec,):
            skip_unsupported_fmt_scale(fmt, scale_cfg)


def operand_fmt(dtype, tensor, gemm):
    """Return a tensor format for one GEMM; strings apply to both."""
    spec = dtype[tensor]
    return spec[gemm] if isinstance(spec, dict) else spec


def rel(got, ref):
    return (got.float() - ref.float()).norm() / ref.float().norm()


def rule(dtype, scale_cfg=None, rounding=None, rotation=None):
    """Build a resolved rule through TrainingConfig.

    Rename `scale_dtype` for config syntax without resolving it again.
    """
    spec = {"enabled": True, "dtype": dict(dtype)}
    if scale_cfg is not None:
        spec["scale"] = {
            **scale_cfg,
            "block_shape": (
                dict(scale_cfg["block_shape"])
                if isinstance(scale_cfg["block_shape"], dict)
                else {
                    tensor: scale_cfg["block_shape"]
                    for tensor in ("weight", "act", "grad_out")
                }
            ),
            "scale_dtype": SCALE_DTYPE_NAMES[scale_cfg["scale_dtype"]],
        }
    if rounding is not None:
        spec["rounding"] = dict(rounding)
    if rotation is not None:
        spec["rotation"] = dict(rotation)
    return TrainingConfig(mixed_precision="no", quantization=spec).quantization[0]
