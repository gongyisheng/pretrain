"""Shared workloads for GEMM backend precision tests."""

from dataclasses import dataclass

import torch


BATCH_SIZE = 16
SEQUENCE_LENGTH = 1024
TOP_K = 8
EXPERT_COUNT = 64
GROUPED_LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")


@dataclass(frozen=True, slots=True)
class GroupedProjection:
    name: str
    k: int
    n: int


GROUPED_PROJECTIONS = (
    GroupedProjection("gate-up", 512, 384),
    GroupedProjection("down", 192, 512),
)


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
