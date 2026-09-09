"""Shared workloads for MM backend precision tests."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.quant.quantize import quantize_operand


@dataclass(frozen=True, slots=True)
class ScaledMMCase:
    """A scaled mm workload; `m` of None takes the default row count."""

    name: str
    k: int
    n: int
    m: int | None = None


@dataclass(frozen=True, slots=True)
class GroupedMMCase:
    """A grouped mm workload: contraction shape plus the per-group row counts.

    `m` is the dense row count, which only the rank-invalid `3d_x_3d` layout needs.
    """

    name: str
    k: int
    n: int
    counts: Sequence[int]
    m: int = 32


@dataclass(frozen=True, slots=True)
class ScaledGroupedMMCase:
    """A scaled grouped mm workload: contraction shape, group counts, dense M."""

    name: str
    k: int
    n: int
    counts: Sequence[int]
    m: int = 32


@dataclass(frozen=True, slots=True)
class QuantFormatCase:
    name: str
    a_format: str
    b_format: str


@dataclass(frozen=True, slots=True)
class QuantScaleCase:
    name: str
    granularity: str
    block_shape: tuple[int, int]
    block_size: int


@dataclass(frozen=True, slots=True)
class OutDtypeCase:
    name: str
    dtype: torch.dtype
    rtol: float
    atol: float


BIAS_CASES = (False, True)

DEFAULT_COUNTS = (72, 88, 104, 120, 136, 152, 168, 184) * 8
RAGGED_COUNTS = (300, 5, 120, 0, 44, 210, 7, 90, 33, 150, 1, 260, 12, 80, 40, 401)

GROUPED_MM_CASES = (
    GroupedMMCase("gate-up", 512, 384, DEFAULT_COUNTS),
    GroupedMMCase("down", 192, 512, DEFAULT_COUNTS),
    GroupedMMCase("odd-n", 64, 51, RAGGED_COUNTS),
    GroupedMMCase("latent-gate-up", 64, 384, RAGGED_COUNTS),
    GroupedMMCase("latent-down", 192, 64, RAGGED_COUNTS),
)

GROUPED_LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")
GROUPED_ERROR_LAYOUTS = ("ragged_n", "3d_x_3d")

SCALED_MM_CASES = (
    ScaledMMCase("attention", 512, 512),
    ScaledMMCase("dense-gate-up", 512, 3072),
    ScaledMMCase("dense-down", 1536, 512),
    ScaledMMCase("moe-gate-up", 512, 384),
    ScaledMMCase("moe-down", 192, 512),
    # M, K and N all odd, with K leaving every scale block width a partial tail
    ScaledMMCase("nonmultiple", 100, 130, m=70),
)

MXFP8_SCALED_MM_CASES = (
    ScaledMMCase("attention", 512, 512),
    ScaledMMCase("dense-gate-up", 512, 3072),
    ScaledMMCase("dense-down", 1536, 512),
    ScaledMMCase("moe-gate-up", 512, 384),
    ScaledMMCase("moe-down", 192, 512),
    ScaledMMCase("tail", 112, 144, m=80),
)

NVFP4_SCALED_MM_CASES = (
    ScaledMMCase("attention", 512, 512, m=128),
    ScaledMMCase("dense-gate-up", 512, 3072, m=128),
    ScaledMMCase("dense-down", 1536, 512, m=128),
    ScaledMMCase("moe-gate-up", 512, 384, m=128),
    ScaledMMCase("moe-down", 192, 512, m=128),
    ScaledMMCase("mn-tail", 80, 50, m=70),
)

NVFP4_BLOCK_SIZES = (16, 32, 48, 64, 128, 256)

QUANT_FORMAT_CASES = (
    QuantFormatCase("int8xint8", "int8", "int8"),
    QuantFormatCase("e4m3xe4m3", "fp8_e4m3", "fp8_e4m3"),
    QuantFormatCase("e5m2xe5m2", "fp8_e5m2", "fp8_e5m2"),
    QuantFormatCase("e5m2xe4m3", "fp8_e5m2", "fp8_e4m3"),
    QuantFormatCase("e4m3xe5m2", "fp8_e4m3", "fp8_e5m2"),
)

MXFP8_FORMAT_CASES = (QuantFormatCase("e4m3xe4m3", "fp8_e4m3", "fp8_e4m3"),)
MXFP8_PLUS_FORMAT_CASES = (
    *MXFP8_FORMAT_CASES,
    QuantFormatCase("e5m2xe5m2", "fp8_e5m2", "fp8_e5m2"),
    QuantFormatCase("e5m2xe4m3", "fp8_e5m2", "fp8_e4m3"),
    QuantFormatCase("e4m3xe5m2", "fp8_e4m3", "fp8_e5m2"),
)

TENSORWISE_SCALE = QuantScaleCase("tensorwise", "tensorwise", (0, 0), 0)
ROWWISE_SCALE = QuantScaleCase("rowwise", "rowwise", (1, 0), 0)
BLOCKWISE1D_8_SCALE = QuantScaleCase("1x8", "blockwise", (1, 8), 8)
BLOCKWISE1D_16_SCALE = QuantScaleCase("1x16", "blockwise", (1, 16), 16)
BLOCKWISE1D_32_SCALE = QuantScaleCase("1x32", "blockwise", (1, 32), 32)
BLOCKWISE1D_48_SCALE = QuantScaleCase("1x48", "blockwise", (1, 48), 48)
BLOCKWISE1D_64_SCALE = QuantScaleCase("1x64", "blockwise", (1, 64), 64)
BLOCKWISE1D_128_SCALE = QuantScaleCase("1x128", "blockwise", (1, 128), 128)
BLOCKWISE2D_16_SCALE = QuantScaleCase("16x16", "blockwise", (16, 16), 16)
BLOCKWISE2D_32_SCALE = QuantScaleCase("32x32", "blockwise", (32, 32), 32)
BLOCKWISE2D_64_SCALE = QuantScaleCase("64x64", "blockwise", (64, 64), 64)
BLOCKWISE2D_128_SCALE = QuantScaleCase("128x128", "blockwise", (128, 128), 128)

QUANT_SCALE_CASES = (
    TENSORWISE_SCALE,
    ROWWISE_SCALE,
    BLOCKWISE1D_16_SCALE,
    BLOCKWISE1D_32_SCALE,
    BLOCKWISE1D_64_SCALE,
    BLOCKWISE1D_128_SCALE,
    BLOCKWISE2D_16_SCALE,
    BLOCKWISE2D_32_SCALE,
    BLOCKWISE2D_64_SCALE,
    BLOCKWISE2D_128_SCALE,
)

QUANT_SCALE_ERROR_CASES = (
    BLOCKWISE1D_8_SCALE,
    BLOCKWISE1D_48_SCALE,
)

MXFP8_SCALE_CASES = (
    BLOCKWISE1D_32_SCALE,
    BLOCKWISE2D_32_SCALE,
)

MXFP8_SCALE_ERROR_CASES = (
    TENSORWISE_SCALE,
    ROWWISE_SCALE,
    BLOCKWISE1D_8_SCALE,
    BLOCKWISE1D_16_SCALE,
    BLOCKWISE1D_48_SCALE,
    BLOCKWISE1D_64_SCALE,
    BLOCKWISE2D_64_SCALE,
)

MXFP8_PLUS_SCALE_CASES = (
    *MXFP8_SCALE_CASES,
    BLOCKWISE1D_64_SCALE,
    BLOCKWISE2D_64_SCALE,
)

MXFP8_PLUS_SCALE_ERROR_CASES = (
    TENSORWISE_SCALE,
    ROWWISE_SCALE,
    BLOCKWISE1D_8_SCALE,
    BLOCKWISE1D_16_SCALE,
    BLOCKWISE1D_48_SCALE,
)

SCALED_GROUPED_MM_CASES = (
    ScaledGroupedMMCase("scaled-grouped", 64, 48, RAGGED_COUNTS),
    ScaledGroupedMMCase("scaled-grouped-tail", 163, 48, RAGGED_COUNTS),
)

NVFP4_SCALED_GROUPED_MM_CASES = (
    ScaledGroupedMMCase("nvfp4-grouped", 192, 48, (16, 32, 0, 32), m=40),
    ScaledGroupedMMCase("nvfp4-grouped-tail", 208, 50, (70, 18, 0, 120), m=70),
)

BF16_OUT = OutDtypeCase("bf16", torch.bfloat16, rtol=2e-2, atol=2e-2)
FP16_OUT = OutDtypeCase("fp16", torch.float16, rtol=2e-3, atol=2e-3)

OUT_DTYPE_CASES = (BF16_OUT, FP16_OUT)


def _pack_e2m1(shape: tuple[int, ...], dim: int, device: str) -> torch.Tensor:
    codes = torch.randint(0, 16, shape, device=device, dtype=torch.uint8)
    axis = dim % len(shape)
    even = torch.arange(0, shape[axis], 2, device=device)
    odd = even + 1
    low = codes.index_select(axis, even)
    high = codes.index_select(axis, odd)
    return low | high << 4


def _nvfp4_scales(shape: tuple[int, ...], device: str) -> torch.Tensor:
    scales = torch.randint(1, 17, shape, device=device).float() / 8
    return scales.to(torch.float8_e4m3fn)


def _make_random_tensor(
    shape: tuple[int, ...],
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=dtype) * 0.1


def _make_grouped_offsets(counts: Sequence[int], device: str) -> torch.Tensor:
    """The cumulative end-offsets torch._grouped_mm partitions a ragged dim by."""
    return torch.tensor(counts, device=device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )


def make_grouped_mm_inputs(
    case: GroupedMMCase,
    layout: str,
    with_bias: bool,
    out_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Operands for one grouped layout, named by which dim `offs` partitions."""
    torch.manual_seed(seed)
    offs = _make_grouped_offsets(case.counts, device)
    expert_count, rows = len(case.counts), sum(case.counts)

    if layout == "ragged_m":  # (rows,K) x (G,K,N) -> (rows,N)
        a = _make_random_tensor((rows, case.k), device, out_dtype)
        b = _make_random_tensor((expert_count, case.k, case.n), device, out_dtype)
    elif layout == "ragged_k":  # (K,rows) x (rows,N) -> (G,K,N)
        a = _make_random_tensor((rows, case.k), device, out_dtype).mT
        b = _make_random_tensor((rows, case.n), device, out_dtype)
    elif layout == "ragged_n":  # (G,K,N) x (N,rows) -> (K,rows)
        a = _make_random_tensor((expert_count, case.k, case.n), device, out_dtype)
        b = _make_random_tensor((case.n, rows), device, out_dtype)
    elif layout == "3d_x_3d":  # (G,M,K) x (G,K,N): the one rank pair no layout defines
        a = _make_random_tensor((expert_count, case.m, case.k), device, out_dtype)
        b = _make_random_tensor((expert_count, case.k, case.n), device, out_dtype)
    else:
        raise ValueError(f"unknown grouped MM layout: {layout}")

    bias = (
        _make_random_tensor((expert_count, case.n), device, out_dtype)
        if with_bias
        else None
    )
    return a, b, offs, bias


def make_scaled_mm_inputs(
    case: ScaledMMCase,
    format: QuantFormatCase | None = None,
    scale: QuantScaleCase | None = None,
    with_bias: bool = False,
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 0,
    with_global_scale: bool = False,
    packed_e2m1: bool = False,
    block_size: int = 16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.dtype,
    int,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    torch.manual_seed(seed)
    rows = case.m if case.m is not None else 8 * 1024
    if packed_e2m1:
        aq = _pack_e2m1((rows, case.k), -1, device)
        bq = _pack_e2m1((case.k, case.n), -2, device)
        scale_blocks = (case.k + block_size - 1) // block_size
        sa = _nvfp4_scales((rows, scale_blocks), device)
        sb = _nvfp4_scales((scale_blocks, case.n), device)
    else:
        if format is None or scale is None:
            raise ValueError("format and scale are required unless packed_e2m1 is set")
        a = _make_random_tensor((rows, case.k), device, out_dtype)
        b = _make_random_tensor((case.n, case.k), device, out_dtype).mT
        scale_config = {
            "granularity": scale.granularity,
            "block_shape": scale.block_shape,
            "scale_dtype": scale_dtype,
            "global_scale": False,
        }
        aq, sa, _ = quantize_operand(a, -1, format.a_format, scale_config)
        bq, sb, _ = quantize_operand(b, -2, format.b_format, scale_config)
        block_size = scale.block_size
    bias = _make_random_tensor((case.n,), device, out_dtype) if with_bias else None
    gsa = torch.tensor([0.5], device=device) if with_global_scale else None
    gsb = torch.tensor([0.25], device=device) if with_global_scale else None
    return (
        aq,
        bq,
        sa,
        sb,
        out_dtype,
        block_size,
        bias,
        gsa,
        gsb,
    )


def make_scaled_grouped_mm_inputs(
    case: ScaledGroupedMMCase,
    layout: str,
    format: QuantFormatCase | None = None,
    scale: QuantScaleCase | None = None,
    with_bias: bool = False,
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 0,
    with_global_scale: bool = False,
    packed_e2m1: bool = False,
    block_size: int = 16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.dtype,
    int,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    torch.manual_seed(seed)
    offs = _make_grouped_offsets(case.counts, device)
    expert_count, rows = len(case.counts), sum(case.counts)
    if not packed_e2m1:
        if format is None or scale is None:
            raise ValueError("format and scale are required unless packed_e2m1 is set")
        scale_config = {
            "granularity": scale.granularity,
            "block_shape": scale.block_shape,
            "scale_dtype": scale_dtype,
            "global_scale": False,
        }
        block_size = scale.block_size

    if packed_e2m1 and layout == "ragged_m":
        aq = _pack_e2m1((rows, case.k), -1, device)
        bq = _pack_e2m1((expert_count, case.k, case.n), -2, device)
        scale_blocks = (case.k + block_size - 1) // block_size
        sa = _nvfp4_scales((rows, scale_blocks), device)
        sb = _nvfp4_scales((expert_count, scale_blocks, case.n), device)
        result = aq, bq, sa, sb

    elif packed_e2m1 and layout == "ragged_k":
        aq = _pack_e2m1((case.m, rows), -1, device)
        bq = _pack_e2m1((rows, case.n), -2, device)
        scale_blocks = sum(
            (count + block_size - 1) // block_size for count in case.counts
        )
        sa = _nvfp4_scales((case.m, scale_blocks), device)
        sb = _nvfp4_scales((scale_blocks, case.n), device)
        result = aq, bq, sa, sb

    elif packed_e2m1 and layout == "ragged_n":
        aq = _pack_e2m1((expert_count, case.m, case.k), -1, device)
        bq = _pack_e2m1((case.k, rows), -2, device)
        scale_blocks = (case.k + block_size - 1) // block_size
        sa = _nvfp4_scales((expert_count, case.m, scale_blocks), device)
        sb = _nvfp4_scales((scale_blocks, rows), device)
        result = aq, bq, sa, sb

    elif layout == "ragged_m":
        aq, sa, _ = quantize_operand(
            _make_random_tensor((rows, case.k), device, out_dtype),
            -1,
            format.a_format,
            scale_config,
        )
        b = torch.stack(
            [
                _make_random_tensor((case.k, case.n), device, out_dtype)
                for _ in range(expert_count)
            ]
        )
        bq, sb, _ = quantize_operand(b, -2, format.b_format, scale_config)
        result = aq, bq, sa, sb

    elif layout == "ragged_k":
        aq, sa, _ = quantize_operand(
            _make_random_tensor((rows, case.m), device, out_dtype),
            -2,
            format.a_format,
            scale_config,
            offs=offs,
            ragged_dim=-2,
        )
        bq, sb, _ = quantize_operand(
            _make_random_tensor((rows, case.n), device, out_dtype),
            -2,
            format.b_format,
            scale_config,
            offs=offs,
            ragged_dim=-2,
        )
        result = aq.mT, bq, sa.mT, sb

    elif layout == "ragged_n":
        a = torch.stack(
            [
                _make_random_tensor((case.m, case.k), device, out_dtype)
                for _ in range(expert_count)
            ]
        )
        aq, sa, _ = quantize_operand(a, -1, format.a_format, scale_config)
        bqT, sbT, _ = quantize_operand(
            _make_random_tensor((rows, case.k), device, out_dtype),
            -1,
            format.b_format,
            scale_config,
            offs=offs,
            ragged_dim=-2,
        )
        result = aq, bqT.mT, sa, sbT.mT

    elif not packed_e2m1 and layout == "3d_x_3d":
        aq, sa, _ = quantize_operand(
            _make_random_tensor((expert_count, case.m, case.k), device, out_dtype),
            -1,
            format.a_format,
            scale_config,
        )
        bq, sb, _ = quantize_operand(
            _make_random_tensor((expert_count, case.k, case.n), device, out_dtype),
            -2,
            format.b_format,
            scale_config,
        )
        result = aq, bq, sa, sb

    else:
        raise ValueError(layout)

    bias = (
        _make_random_tensor((expert_count, case.n), device, out_dtype)
        if with_bias
        else None
    )
    gsa = (
        torch.arange(1, expert_count + 1, device=device).float() / 4
        if with_global_scale
        else None
    )
    gsb = (
        torch.arange(expert_count, 0, -1, device=device).float() / 4
        if with_global_scale
        else None
    )
    return *result, offs, out_dtype, block_size, bias, gsa, gsb
