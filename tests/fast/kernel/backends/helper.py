"""Shared workloads for GEMM backend precision tests."""

from dataclasses import dataclass

import torch


BATCH_SIZE = 8
SEQUENCE_LENGTH = 1024
TOP_K = 8
EXPERT_COUNT = 64
GROUPED_LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")


@dataclass(frozen=True, slots=True)
class GroupedProjection:
    name: str
    k: int
    n: int


@dataclass(frozen=True, slots=True)
class DenseWorkload:
    name: str
    n: int
    k: int


@dataclass(frozen=True, slots=True)
class FormatPair:
    name: str
    a_format: str
    b_format: str
    rtol: float
    atol: float


@dataclass(frozen=True, slots=True)
class ScalingCase:
    name: str
    granularity: str
    block_shape: tuple[int, int]
    block_size: int


GROUPED_PROJECTIONS = (
    GroupedProjection("gate-up", 512, 384),
    GroupedProjection("down", 192, 512),
)

DENSE_WORKLOADS = (
    DenseWorkload("attention", 512, 512),
    DenseWorkload("dense-gate-up", 3072, 512),
    DenseWorkload("dense-down", 512, 1536),
    DenseWorkload("moe-gate-up", 384, 512),
    DenseWorkload("moe-down", 512, 192),
)

FORMAT_PAIRS = (
    FormatPair("int8xint8", "int8", "int8", 2e-2, 2e-2),
    FormatPair("e4m3xe4m3", "fp8_e4m3", "fp8_e4m3", 2e-2, 2e-2),
    FormatPair("e5m2xe5m2", "fp8_e5m2", "fp8_e5m2", 5e-2, 5e-2),
    FormatPair("e5m2xe4m3", "fp8_e5m2", "fp8_e4m3", 5e-2, 5e-2),
    FormatPair("e4m3xe5m2", "fp8_e4m3", "fp8_e5m2", 5e-2, 5e-2),
)

SCALING_CASES = (
    ScalingCase("tensorwise", "tensorwise", (0, 0), 0),
    ScalingCase("rowwise", "rowwise", (1, 0), 0),
    *(
        ScalingCase(f"1x{block_size}", "blockwise", (1, block_size), block_size)
        for block_size in (16, 32, 64, 128)
    ),
    *(
        ScalingCase(
            f"{block_size}x{block_size}",
            "blockwise",
            (block_size, block_size),
            block_size,
        )
        for block_size in (16, 32, 64, 128)
    ),
)

BIAS_CASES = (False, True)


def grouped_counts(batch_size: int = BATCH_SIZE) -> list[int]:
    pattern = [72, 88, 104, 120, 136, 152, 168, 184]
    return [batch_size * count for count in pattern] * 8


def make_grouped_inputs(
    projection: GroupedProjection,
    layout: str,
    with_bias: bool,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    torch.manual_seed(seed)
    counts = grouped_counts(batch_size)
    rows = sum(counts)
    offs = torch.tensor(counts, device=device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )

    def rand(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1

    if layout == "ragged_m":
        a = rand((rows, projection.k))
        b = rand((EXPERT_COUNT, projection.k, projection.n))
    elif layout == "ragged_k":
        a = rand((rows, projection.k)).mT
        b = rand((rows, projection.n))
    elif layout == "ragged_n":
        a = rand((EXPERT_COUNT, projection.k, projection.n))
        b = rand((projection.n, rows))
    else:
        raise ValueError(f"unknown grouped GEMM layout: {layout}")

    bias = rand((EXPERT_COUNT, projection.n)) if with_bias else None
    return a, b, offs, bias


def make_scaled_gemm_inputs(
    workload: DenseWorkload,
    format_pair: FormatPair,
    scaling_case: ScalingCase,
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
    a = torch.randn(rows, workload.k, device=device, dtype=torch.bfloat16) * 0.1
    b = torch.randn(workload.k, workload.n, device=device, dtype=torch.bfloat16) * 0.1
    scaling = {
        "granularity": scaling_case.granularity,
        "block_shape": scaling_case.block_shape,
        "scale_dtype": torch.float8_e8m0fnu if mx_scales else torch.float32,
    }
    aq, sa = quantize_operand(a, -1, format_pair.a_format, scaling)
    bq, sb = quantize_operand(b, -2, format_pair.b_format, scaling)
    bias = (
        torch.randn(workload.n, device=device, dtype=torch.bfloat16) * 0.1
        if with_bias
        else None
    )
    return (
        aq,
        bq,
        sa,
        sb,
        torch.bfloat16,
        scaling_case.block_size,
        bias,
    )
