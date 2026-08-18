"""Shared workloads for GEMM backend precision tests."""

from dataclasses import dataclass

import torch


BATCH_SIZE = 8
SEQUENCE_LENGTH = 1024
TOP_K = 8
EXPERT_COUNT = 64
GROUPED_LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")


@dataclass(frozen=True, slots=True)
class GemmShape:
    name: str
    k: int
    n: int


@dataclass(frozen=True, slots=True)
class QuantFormatCase:
    name: str
    a_format: str
    b_format: str
    rtol: float
    atol: float


@dataclass(frozen=True, slots=True)
class QuantScaleCase:
    name: str
    granularity: str
    block_shape: tuple[int, int]
    block_size: int


GROUPED_GEMM_SHAPES = (
    GemmShape("gate-up", 512, 384),
    GemmShape("down", 192, 512),
)

DENSE_GEMM_SHAPES = (
    GemmShape("attention", 512, 512),
    GemmShape("dense-gate-up", 512, 3072),
    GemmShape("dense-down", 1536, 512),
    GemmShape("moe-gate-up", 512, 384),
    GemmShape("moe-down", 192, 512),
)

QUANT_FORMAT_CASES = (
    QuantFormatCase("int8xint8", "int8", "int8", 2e-2, 2e-2),
    QuantFormatCase("e4m3xe4m3", "fp8_e4m3", "fp8_e4m3", 2e-2, 2e-2),
    QuantFormatCase("e5m2xe5m2", "fp8_e5m2", "fp8_e5m2", 5e-2, 5e-2),
    QuantFormatCase("e5m2xe4m3", "fp8_e5m2", "fp8_e4m3", 5e-2, 5e-2),
    QuantFormatCase("e4m3xe5m2", "fp8_e4m3", "fp8_e5m2", 5e-2, 5e-2),
)

QUANT_SCALE_CASES = (
    QuantScaleCase("tensorwise", "tensorwise", (0, 0), 0),
    QuantScaleCase("rowwise", "rowwise", (1, 0), 0),
    *(
        QuantScaleCase(f"1x{block_size}", "blockwise", (1, block_size), block_size)
        for block_size in (16, 32, 64, 128)
    ),
    *(
        QuantScaleCase(
            f"{block_size}x{block_size}",
            "blockwise",
            (block_size, block_size),
            block_size,
        )
        for block_size in (16, 32, 64, 128)
    ),
)

BIAS_CASES = (False, True)


def _make_random_tensor(
    shape: tuple[int, ...],
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=dtype) * 0.1


def make_grouped_mm_inputs(
    projection: GemmShape,
    layout: str,
    with_bias: bool,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    torch.manual_seed(seed)
    counts = [
        batch_size * count for count in (72, 88, 104, 120, 136, 152, 168, 184)
    ] * 8
    rows = sum(counts)
    offs = torch.tensor(counts, device=device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )

    if layout == "ragged_m":
        a = _make_random_tensor((rows, projection.k), device, dtype)
        b = _make_random_tensor(
            (EXPERT_COUNT, projection.k, projection.n), device, dtype
        )
    elif layout == "ragged_k":
        a = _make_random_tensor((rows, projection.k), device, dtype).mT
        b = _make_random_tensor((rows, projection.n), device, dtype)
    elif layout == "ragged_n":
        a = _make_random_tensor(
            (EXPERT_COUNT, projection.k, projection.n), device, dtype
        )
        b = _make_random_tensor((projection.n, rows), device, dtype)
    else:
        raise ValueError(f"unknown grouped GEMM layout: {layout}")

    bias = (
        _make_random_tensor((EXPERT_COUNT, projection.n), device, dtype)
        if with_bias
        else None
    )
    return a, b, offs, bias


def make_scaled_mm_inputs(
    workload: GemmShape,
    format_pair: QuantFormatCase,
    scaling_case: QuantScaleCase,
    mx_scales: bool,
    with_bias: bool,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.dtype,
    int,
    torch.Tensor | None,
]:
    from src.quant.quantize import quantize_operand

    torch.manual_seed(seed)
    rows = batch_size * SEQUENCE_LENGTH
    a = _make_random_tensor((rows, workload.k), device)
    b = _make_random_tensor((workload.k, workload.n), device)
    scaling = {
        "granularity": scaling_case.granularity,
        "block_shape": scaling_case.block_shape,
        "scale_dtype": torch.float8_e8m0fnu if mx_scales else torch.float32,
    }
    aq, sa = quantize_operand(a, -1, format_pair.a_format, scaling)
    bq, sb = quantize_operand(b, -2, format_pair.b_format, scaling)
    bias = _make_random_tensor((workload.n,), device) if with_bias else None
    return (
        aq,
        bq,
        sa,
        sb,
        torch.bfloat16,
        scaling_case.block_size,
        bias,
    )


def make_scaled_grouped_mm_inputs(
    layout,
    counts,
    gran,
    bs,
    fmt,
    M=32,
    K=64,
    N=48,
    seed=0,
    scale_dtype=torch.float32,
):
    """Quantized operands and scales for one ragged scaled grouped GEMM layout.

    Scales mirror each operand's axis order, replacing the contraction axis with the
    scale-block axis. A zero `bs` (row/tensorwise) uses one block per contraction
    segment.
    """
    from src.quant.quantize import quantize_operand

    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    scaling = {
        "granularity": gran,
        "block_shape": (1, bs) if bs else (0, 0),
        "scale_dtype": scale_dtype,
    }

    if layout == "ragged_m":
        aq, sa = quantize_operand(_make_random_tensor((R, K), "cuda"), -1, fmt, scaling)
        b = torch.stack([_make_random_tensor((K, N), "cuda") for _ in range(E)])
        bq, sb = quantize_operand(b, -2, fmt, scaling)
        return aq, bq, sa, sb, offs, bs

    if layout == "ragged_k":
        aq, sa = quantize_operand(
            _make_random_tensor((R, M), "cuda"),
            -2,
            fmt,
            scaling,
            offs=offs,
            ragged_dim=-2,
        )
        bq, sb = quantize_operand(
            _make_random_tensor((R, N), "cuda"),
            -2,
            fmt,
            scaling,
            offs=offs,
            ragged_dim=-2,
        )
        return aq.mT, bq, sa.mT, sb, offs, bs

    if layout == "ragged_n":
        a = torch.stack([_make_random_tensor((M, K), "cuda") for _ in range(E)])
        aq, sa = quantize_operand(a, -1, fmt, scaling)
        bqT, sbT = quantize_operand(
            _make_random_tensor((R, K), "cuda"),
            -1,
            fmt,
            scaling,
            offs=offs,
            ragged_dim=-2,
        )
        return aq, bqT.mT, sa, sbT.mT, offs, bs

    raise ValueError(layout)
