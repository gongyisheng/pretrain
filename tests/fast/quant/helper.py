"""Shared formats, scale configs, and reference helpers for quantization tests."""

import pytest
import torch

from src.quant.constants import QUANT_PASSTHROUGH, _FP8_FORMATS, _INT8_FORMATS
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.utils import is_fp8, is_int8s, is_quantized
from src.utils.config import TrainingConfig


def fp8_capable():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not fp8_capable(), reason="fp8 needs SM >= 8.9")

E4M3 = "fp8_e4m3"
FP8_FORMATS = sorted(_FP8_FORMATS)
INT_FORMATS = sorted(_INT8_FORMATS)
ALL_QUANT_FORMATS = FP8_FORMATS + INT_FORMATS
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


def has_passthrough(dtype):
    """Whether any of a cell's slots is left unquantized, per-GEMM specs included. Such
    a GEMM has no fused op and falls back to dequant-and-matmul, which is a different
    accuracy regime from a fully quantized cell."""
    return any(
        not is_quantized(fmt)
        for spec in dtype.values()
        for fmt in (spec.values() if isinstance(spec, dict) else (spec,))
    )


def scale_of(granularity, block_shape=(0, 0), scale_dtype=torch.float32):
    return {
        "granularity": granularity,
        "block_shape": block_shape,
        "scale_dtype": scale_dtype,
    }


TENSORWISE = scale_of("tensorwise")
ROWWISE = scale_of("rowwise")
# Blockwise contract extents are multiples of 16; outer extents are 1 or square.
BLOCKWISE1D_16 = scale_of("blockwise", (1, 16))
BLOCKWISE1D_32 = scale_of("blockwise", (1, 32))
BLOCKWISE1D_64 = scale_of("blockwise", (1, 64))
BLOCKWISE1D_128 = scale_of("blockwise", (1, 128))
BLOCKWISE2D_16 = scale_of("blockwise", (16, 16))
BLOCKWISE2D_32 = scale_of("blockwise", (32, 32))
BLOCKWISE2D_64 = scale_of("blockwise", (64, 64))
BLOCKWISE2D_128 = scale_of("blockwise", (128, 128))
# E8M0 MX scales require a multiple-of-32 extent; 2D-64 also covers `rep_k == 2`.
MXFP8 = scale_of("blockwise", (1, 32), torch.float8_e8m0fnu)
BLOCKWISE2D_64_E8M0 = scale_of("blockwise", (64, 64), torch.float8_e8m0fnu)


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
    MXFP8,
    BLOCKWISE2D_64_E8M0,
]

SCALE_DTYPE_NAMES = {torch.float32: "fp32", torch.float8_e8m0fnu: "fp8_e8m0"}


def roundtrip(x, contract_dim, fmt, scale_cfg):
    """Return the dequantized operand used by quantized GEMM."""
    if not is_quantized(fmt):
        return x
    xq, scale = quantize_operand(x, contract_dim, fmt, scale_cfg)
    return dequantize_operand(xq, scale, contract_dim, scale_cfg)


def fused_op_exists(a_fmt, b_fmt):
    """Whether the pair has a fused scaled GEMM: both quantized, same family. Anything
    else -- cross-family, or one side unquantized -- takes the dequant fallback."""
    return (is_fp8(a_fmt) and is_fp8(b_fmt)) or (is_int8s(a_fmt) and is_int8s(b_fmt))


def mm_ref(a, a_fmt, b, b_fmt, scale_cfg):
    """The GEMM quantized_mm actually performs, dequantized where that path dequantizes:
    a fused kernel works in its fp32 accumulator, while the fallback rounds the
    dequantized operand back to the operands' own dtype first. Getting that point wrong
    costs a bf16 rounding, which reads as ~2e-3 of error that is not there.

    Returns fp32: both paths accumulate in fp32 and round once at the output, so the
    caller adds any bias here and casts the sum, rather than rounding twice.
    """
    dtype = torch.float32 if fused_op_exists(a_fmt, b_fmt) else a.dtype
    return (
        roundtrip(a, -1, a_fmt, scale_cfg).to(dtype).float()
        @ roundtrip(b, -2, b_fmt, scale_cfg).to(dtype).float()
    )


def has_int(dtype):
    """Whether any cell slot names an int format; e8m0 scales require fp8 elements."""
    return any(
        is_int8s(fmt)
        for spec in dtype.values()
        for fmt in (spec.values() if isinstance(spec, dict) else (spec,))
    )


def operand_fmt(dtype, tensor, gemm):
    """Return a tensor's format for one GEMM; strings apply to both GEMMs."""
    spec = dtype[tensor]
    return spec[gemm] if isinstance(spec, dict) else spec


def rel(got, ref):
    return (got.float() - ref.float()).norm() / ref.float().norm()


def rule(dtype, scale_cfg=None, rounding=None):
    """Build a resolved rule, filling omitted dtype slots through TrainingConfig.

    Convert `scale_cfg` to config syntax by renaming `scale_dtype` without resolving it
    again, keeping comparisons independent.
    """
    spec = {"enabled": True, "dtype": dict(dtype)}
    if scale_cfg is not None:
        spec["scale"] = {
            **scale_cfg,
            "scale_dtype": SCALE_DTYPE_NAMES[scale_cfg["scale_dtype"]],
        }
    if rounding is not None:
        spec["rounding"] = dict(rounding)
    return TrainingConfig(mixed_precision="no", quantization=spec).quantization[0]
