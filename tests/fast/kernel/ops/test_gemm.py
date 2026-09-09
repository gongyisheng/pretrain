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
    nvfp4_scaled_grouped_mm,
    nvfp4_scaled_mm,
)

E, M, K, N = 4, 32, 64, 48

SCALED_MM_FUNCS = (
    int8_scaled_mm,
    fp8_scaled_mm,
    mxfp8_scaled_mm,
    nvfp4_scaled_mm,
)
SCALED_GROUPED_MM_FUNCS = (
    int8_scaled_grouped_mm,
    fp8_scaled_grouped_mm,
    mxfp8_scaled_grouped_mm,
    nvfp4_scaled_grouped_mm,
)
OP_IDS = {
    int8_scaled_mm: "int8",
    fp8_scaled_mm: "fp8",
    mxfp8_scaled_mm: "mxfp8",
    nvfp4_scaled_mm: "nvfp4",
    int8_scaled_grouped_mm: "int8",
    fp8_scaled_grouped_mm: "fp8",
    mxfp8_scaled_grouped_mm: "mxfp8",
    nvfp4_scaled_grouped_mm: "nvfp4",
}
ZERO_BLOCK_OPS = ("int8", "fp8")
MXFP8_OPS = ("mxfp8",)
NVFP4_OPS = ("nvfp4",)

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
    """A workload: the layout and shapes to override."""

    name: str
    layout: str
    block_size: int | None = None
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    non_contiguous_offs: bool = False
    a_dtype: torch.dtype | None = None
    b_dtype: torch.dtype | None = None
    scale_dtype: torch.dtype | None = None
    # Global scales to attach, by argument name, as {"gsa": shape, ...}.
    global_scale_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    global_scale_dtype: torch.dtype = torch.float32
    applicable_ops: tuple[str, ...] = ()
    offs_delta: tuple[int, int] | None = None
    error_type: type[Exception] = ValueError
    expected_shape: tuple[int, ...] | None = None


def make_inputs(case, op=None):
    """Build zero-filled operands and unit scales for `case`."""
    mxfp8 = op in (mxfp8_scaled_mm, mxfp8_scaled_grouped_mm)
    fp8 = op in (fp8_scaled_mm, fp8_scaled_grouped_mm) or mxfp8
    packed_e2m1 = op in (nvfp4_scaled_mm, nvfp4_scaled_grouped_mm)
    block_size = (
        case.block_size
        if case.block_size is not None
        else (32 if mxfp8 else 16 if packed_e2m1 else 0)
    )
    blocks = -(-K // block_size) if block_size else 1
    shapes = LAYOUT_SHAPES[case.layout](blocks) | case.shapes
    scale_dtype = (
        case.scale_dtype
        if case.scale_dtype is not None
        else torch.float8_e4m3fn
        if packed_e2m1
        else torch.float8_e8m0fnu
        if mxfp8
        else torch.float32
    )
    a_dtype = (
        case.a_dtype
        if case.a_dtype is not None
        else torch.uint8
        if packed_e2m1
        else torch.float8_e4m3fn
        if fp8
        else torch.int8
    )
    b_dtype = (
        case.b_dtype
        if case.b_dtype is not None
        else torch.uint8
        if packed_e2m1
        else torch.float8_e4m3fn
        if fp8
        else torch.int8
    )
    if packed_e2m1:
        shapes["aq"] = (*shapes["aq"][:-1], shapes["aq"][-1] // 2)
        shapes["bq"] = (*shapes["bq"][:-2], shapes["bq"][-2] // 2, shapes["bq"][-1])
    global_scale_shapes = case.global_scale_shapes
    if packed_e2m1 and not global_scale_shapes:
        groups = 1 if case.layout == "mm" else E
        global_scale_shapes = {"gsa": (groups,), "gsb": (groups,)}
    args = {
        "aq": torch.zeros(shapes["aq"], dtype=a_dtype),
        "bq": torch.zeros(shapes["bq"], dtype=b_dtype),
        "sa": torch.ones(shapes["sa"], dtype=scale_dtype),
        "sb": torch.ones(shapes["sb"], dtype=scale_dtype),
        "out_dtype": torch.bfloat16,
        "block_size": block_size,
    }
    args |= {
        name: torch.ones(shape, dtype=case.global_scale_dtype)
        for name, shape in global_scale_shapes.items()
    }
    if case.layout == "mm":
        return args
    offs = torch.arange(1, E + 1, dtype=torch.int32) * OFFS_STRIDE[case.layout]
    if case.non_contiguous_offs:
        wide = torch.zeros(E * 2, dtype=torch.int32)
        wide[::2] = offs
        offs = wide[::2]
    if case.offs_delta is not None:
        index, delta = case.offs_delta
        offs[index] += delta
    args["offs"] = offs
    return args


# fmt: off
# Do not remove: this keeps each case on one line for readability.
SCALED_MM_ERROR_CASES = (
    MMCase("contraction", "mm", shapes={"bq": (K + 32, N)}),
    MMCase("sa_blocks", "mm", block_size=32, shapes={"sa": (M, 4)}),
    MMCase("sb_blocks", "mm", block_size=32, shapes={"sb": (4, N)}),
    # A dense call is one group, so either paired global scale with another length is wrong.
    MMCase("gsa_shape", "mm", block_size=32, global_scale_shapes={"gsa": (2,), "gsb": (1,)}),
    MMCase("gsb_shape", "mm", block_size=32, global_scale_shapes={"gsa": (1,), "gsb": (2,)}),
    # One operand's scale alone: Triton's single HAS_GLOBAL_SCALE constexpr cannot express it.
    MMCase("gsa_alone", "mm", block_size=32, global_scale_shapes={"gsa": (1,)}),
    MMCase("gsa_dtype", "mm", block_size=32, global_scale_shapes={"gsa": (1,), "gsb": (1,)}, global_scale_dtype=torch.float64),
    MMCase("mxfp8_block_size_0", "mm", block_size=0, applicable_ops=MXFP8_OPS),
    MMCase("nvfp4_aq_dtype", "mm", block_size=16, a_dtype=torch.int8, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_bq_dtype", "mm", block_size=16, b_dtype=torch.int8, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_e8m0_scale", "mm", block_size=16, scale_dtype=torch.float8_e8m0fnu, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_block_size_0", "mm", block_size=0, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_block_size_8", "mm", block_size=8, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_block_size_17", "mm", block_size=17, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_logical_k", "mm", block_size=16, shapes={"aq": (M, K - 2), "bq": (K - 2, N)}, applicable_ops=NVFP4_OPS),
)
# fmt: on

# fmt: off
# Do not remove: this keeps each case on one line for readability.
SCALED_GROUPED_MM_ERROR_CASES = (
    MMCase("offs_non_contiguous", "ragged_m", non_contiguous_offs=True),
    MMCase("contraction", "ragged_m", shapes={"bq": (E, K + 32, N)}),
    MMCase("ragged_m_sa_blocks", "ragged_m", block_size=32, shapes={"sa": (M, 4)}),
    MMCase("ragged_m_sb_shape", "ragged_m", shapes={"sb": (E, 1, N + 1)}),
    MMCase("ragged_n_sa_shape", "ragged_n", shapes={"sa": (E, M, 3)}),
    MMCase("ragged_n_sb_shape", "ragged_n", shapes={"sb": (1, N + 1)}),
    MMCase("ragged_k_sa_rows", "ragged_k", shapes={"sa": (M + 1, E)}),
    MMCase("ragged_k_sb_cols", "ragged_k", shapes={"sb": (E, N + 1)}),
    MMCase("ragged_k_scale_pair", "ragged_k", shapes={"sa": (M, E + 1)}),
    # block_size 0 means one scale per group, so blocks must equal the offs groups.
    MMCase("ragged_k_offs_groups", "ragged_k", block_size=0, shapes={"sa": (M, E + 1), "sb": (E + 1, N)}, applicable_ops=ZERO_BLOCK_OPS),
    MMCase("gsa_shape", "ragged_m", global_scale_shapes={"gsa": (E + 1,)}),
    MMCase("gsa_alone", "ragged_m", global_scale_shapes={"gsa": (E,)}),
    MMCase("mxfp8_block_size_0", "ragged_m", block_size=0, applicable_ops=MXFP8_OPS),
    MMCase("nvfp4_e8m0_scale", "ragged_m", block_size=16, scale_dtype=torch.float8_e8m0fnu, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_block_size_0", "ragged_m", block_size=0, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_block_size_8", "ragged_m", block_size=8, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_block_size_17", "ragged_m", block_size=17, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_logical_k", "ragged_m", block_size=16, shapes={"aq": (M, K - 2), "bq": (E, K - 2, N)}, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_odd_ragged_k_offset", "ragged_k", block_size=16, offs_delta=(0, -1), error_type=AssertionError, applicable_ops=NVFP4_OPS),
    MMCase("nvfp4_ragged_k_end", "ragged_k", block_size=16, offs_delta=(-1, -2), error_type=AssertionError, applicable_ops=NVFP4_OPS),
)
# fmt: on

GROUPED_MM_ERROR_CASES = (
    MMCase("offs_non_contiguous", "ragged_m", block_size=0, non_contiguous_offs=True),
    MMCase("contraction", "ragged_m", shapes={"bq": (E, K + 32, N)}),
    MMCase("mixed_dtype", "ragged_m", a_dtype=torch.bfloat16, b_dtype=torch.float16),
)
SCALED_GROUPED_MM_OUTPUT_CASES = (
    MMCase("ragged_m", "ragged_m", expected_shape=(M, N)),
    MMCase("ragged_k", "ragged_k", expected_shape=(E, M, N)),
)
SCALED_MM_OUTPUT_CASES = (
    MMCase(
        "nvfp4_float32_scale",
        "mm",
        block_size=16,
        scale_dtype=torch.float32,
        applicable_ops=NVFP4_OPS,
    ),
)


def operation_case_params(ops, cases):
    """Expand cases across their applicable operations."""
    return tuple(
        pytest.param(op, case, id=f"{OP_IDS[op]}-{case.name}")
        for op in ops
        for case in cases
        if not case.applicable_ops or OP_IDS[op] in case.applicable_ops
    )


@pytest.mark.parametrize("case", GROUPED_MM_ERROR_CASES)
def test_grouped_mm_raise_error(case):
    args = make_inputs(case)
    with pytest.raises(ValueError):
        grouped_mm(args["aq"], args["bq"], args["offs"])


@pytest.mark.parametrize(
    ("op", "case"),
    operation_case_params(SCALED_MM_FUNCS, SCALED_MM_ERROR_CASES),
)
def test_scaled_mm_raise_error(op, case):
    with pytest.raises(ValueError):
        op(**make_inputs(case, op))


@pytest.mark.parametrize(
    ("op", "case"),
    operation_case_params(SCALED_GROUPED_MM_FUNCS, SCALED_GROUPED_MM_ERROR_CASES),
)
def test_scaled_grouped_mm_raise_error(op, case):
    with pytest.raises(case.error_type):
        op(**make_inputs(case, op))


@pytest.mark.parametrize(
    ("op", "case"),
    operation_case_params(SCALED_MM_FUNCS, SCALED_MM_OUTPUT_CASES),
)
def test_scaled_mm_output_shape(op, case):
    assert op(**make_inputs(case, op)).shape == (M, N)


@pytest.mark.parametrize(
    ("op", "case"),
    operation_case_params(SCALED_GROUPED_MM_FUNCS, SCALED_GROUPED_MM_OUTPUT_CASES),
)
def test_scaled_grouped_mm_output_shape(op, case):
    assert op(**make_inputs(case, op)).shape == case.expected_shape
