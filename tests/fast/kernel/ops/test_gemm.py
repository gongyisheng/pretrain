"""Shape and argument validation performed by the gemm ops before dispatch."""

from dataclasses import dataclass, field

import pytest
import torch

from src.kernel.ops.gemm import (
    fp8_scaled_grouped_mm,
    fp8_scaled_mm,
    grouped_mm,
    int8_scaled_grouped_mm,
    int8_scaled_mm,
    mxfp8_scaled_grouped_mm,
    mxfp8_scaled_mm,
)

E, M, K, N = 4, 32, 64, 48

SCALED_MM_FUNCS = (
    int8_scaled_mm,
    fp8_scaled_mm,
    mxfp8_scaled_mm,
)
SCALED_GROUPED_MM_FUNCS = (
    int8_scaled_grouped_mm,
    fp8_scaled_grouped_mm,
    mxfp8_scaled_grouped_mm,
)
OP_IDS = ["int8", "fp8", "mxfp8"]

# Default operand and scale shapes per layout, given the K block count.
LAYOUT_SHAPES = {
    "mm": lambda blocks: {
        "aq": (M, K),
        "bq": (K, N),
        "sa": (M, blocks),
        "sb": (blocks, N),
    },
    "ragged_m": lambda blocks: {
        "aq": (M, K),
        "bq": (E, K, N),
        "sa": (M, blocks),
        "sb": (E, blocks, N),
    },
    "ragged_n": lambda blocks: {
        "aq": (E, M, K),
        "bq": (K, N),
        "sa": (E, M, blocks),
        "sb": (blocks, N),
    },
    # Ragged K carries one scale block per group instead of one per dense K block.
    "ragged_k": lambda blocks: {
        "aq": (M, K),
        "bq": (K, N),
        "sa": (M, E),
        "sb": (E, N),
    },
}
OFFS_STRIDE = {"ragged_m": M // E, "ragged_n": N // E, "ragged_k": K // E}


@dataclass(frozen=True, slots=True)
class MMCase:
    """A workload: the layout, the shapes to override, and the expected message."""

    name: str
    layout: str
    match: str = ""
    block_size: int = 0
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    non_contiguous_offs: bool = False


def make_inputs(case):
    """Build zero-filled operands and unit scales for `case`."""
    blocks = -(-K // case.block_size) if case.block_size else 1
    shapes = LAYOUT_SHAPES[case.layout](blocks) | case.shapes
    args = {
        "aq": torch.zeros(shapes["aq"], dtype=torch.int8),
        "bq": torch.zeros(shapes["bq"], dtype=torch.int8),
        "sa": torch.ones(shapes["sa"]),
        "sb": torch.ones(shapes["sb"]),
        "out_dtype": torch.bfloat16,
        "block_size": case.block_size,
    }
    if case.layout == "mm":
        return args
    offs = torch.arange(1, E + 1, dtype=torch.int32) * OFFS_STRIDE[case.layout]
    if case.non_contiguous_offs:
        wide = torch.zeros(E * 2, dtype=torch.int32)
        wide[::2] = offs
        offs = wide[::2]
    args["offs"] = offs
    return args


SCALED_MM_ERROR_CASES = (
    MMCase("contraction", "mm", "contraction mismatch", shapes={"bq": (K + 32, N)}),
    MMCase("sa_blocks", "mm", "sa block count", 32, {"sa": (M, 4)}),
    MMCase("sb_blocks", "mm", "sb block count", 32, {"sb": (4, N)}),
)

SCALED_GROUPED_MM_ERROR_CASES = (
    MMCase("offs_non_contiguous", "ragged_m", "offs must be contiguous", 0, {}, True),
    MMCase(
        "contraction", "ragged_m", "contraction mismatch", shapes={"bq": (E, K + 32, N)}
    ),
    MMCase("ragged_m_sa_blocks", "ragged_m", "sa block count", 32, {"sa": (M, 4)}),
    MMCase("ragged_m_sb_shape", "ragged_m", "sb block count", 0, {"sb": (E, 1, N + 1)}),
    MMCase("ragged_n_sa_shape", "ragged_n", "sa block count", 0, {"sa": (E, M, 2)}),
    MMCase("ragged_n_sb_shape", "ragged_n", "sb block count", 0, {"sb": (1, N + 1)}),
    MMCase("ragged_k_sa_rows", "ragged_k", "sa rows", 0, {"sa": (M + 1, E)}),
    MMCase("ragged_k_sb_cols", "ragged_k", "sb cols", 0, {"sb": (E, N + 1)}),
    MMCase("ragged_k_scale_pair", "ragged_k", "sa block count", 0, {"sa": (M, E + 1)}),
    # block_size 0 means one scale per group, so blocks must equal the offs groups.
    MMCase(
        "ragged_k_offs_groups",
        "ragged_k",
        "offs groups",
        0,
        {"sa": (M, E + 1), "sb": (E + 1, N)},
    ),
)

GROUPED_MM_ERROR_CASES = (
    MMCase("offs_non_contiguous", "ragged_m", "offs must be contiguous", 0, {}, True),
    MMCase(
        "contraction", "ragged_m", "contraction mismatch", shapes={"bq": (E, K + 32, N)}
    ),
)

OUTPUT_SHAPE_CASES = (
    (MMCase("ragged_m", "ragged_m"), (M, N)),
    (MMCase("ragged_k", "ragged_k"), (E, M, N)),
)


@pytest.mark.parametrize("op", SCALED_MM_FUNCS, ids=OP_IDS)
@pytest.mark.parametrize("case", SCALED_MM_ERROR_CASES, ids=lambda case: case.name)
def test_scaled_mm_raise_error(op, case):
    with pytest.raises(ValueError, match=case.match):
        op(**make_inputs(case))


@pytest.mark.parametrize("op", SCALED_GROUPED_MM_FUNCS, ids=OP_IDS)
@pytest.mark.parametrize(
    "case", SCALED_GROUPED_MM_ERROR_CASES, ids=lambda case: case.name
)
def test_scaled_grouped_mm_raise_error(op, case):
    with pytest.raises(ValueError, match=case.match):
        op(**make_inputs(case))


@pytest.mark.parametrize("case", GROUPED_MM_ERROR_CASES, ids=lambda case: case.name)
def test_grouped_mm_raise_error(case):
    args = make_inputs(case)
    with pytest.raises(ValueError, match=case.match):
        grouped_mm(args["aq"], args["bq"], args["offs"])


@pytest.mark.parametrize(
    ("case", "shape"), OUTPUT_SHAPE_CASES, ids=lambda arg: getattr(arg, "name", None)
)
def test_scaled_grouped_mm_output_shape(case, shape):
    assert int8_scaled_grouped_mm(**make_inputs(case)).shape == shape
