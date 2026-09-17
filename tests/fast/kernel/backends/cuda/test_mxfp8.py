"""CUDA MXFP8 quantization contracts."""

from math import prod

import pytest
import torch

from src.kernel.ops import dequantize_dense, quantize_mxfp8
from tests.fast.kernel.backends.cuda._quantize_test_utils import (
    LARGE_OFFSET_CASES,
    large_offset_source,
)
from tests.fast.helper import cuda_only, cuda_sm89_or_newer


DTYPES = (torch.float32, torch.float16, torch.bfloat16)
CONTRACT_DIMS = (-2, -1)
BLOCK_SHAPES = (
    (1, 16),
    (1, 32),
    (1, 64),
    (1, 128),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
)
TAIL_SHAPES = ((130, 259), (2, 130, 259))
CONTIGUOUS_B32_SHAPES = ((33, 64), (2, 33, 64))
ALIGNED_SQUARE32_SHAPES = ((64, 96), (2, 64, 96))
LARGE_BATCH_SQUARE32_SHAPES = ((65536, 1, 2),)
SHAPES = TAIL_SHAPES + CONTIGUOUS_B32_SHAPES + ALIGNED_SQUARE32_SHAPES
NARROW_SHAPES = ((3, 17), (2, 3, 17))
LAYOUTS = ("dense", "strided", "transposed", "offset")
STAT_SHAPES = ((35, 65), (3, 17), (0, 32), (32, 0))
OUTPUT_LAYOUTS = ("row_major", "column_major")
CUDA_GRAPH_SHAPES = ((130, 256), (130, 259))
COMPILE_SHAPES = ((130, 259), (33, 64))
INDEX_STRIDE_CASES = (
    ((1, 64), (((1 << 31) - 1) // 31, 1), -1),
    ((1, 64), (((1 << 31) - 1) // 31 + 1, 1), -1),
    ((1, 64), ((1 << 31) - 1, 1), -1),
    ((1, 64), (1 << 31, 1), -1),
    ((1, 64), ((1 << 60) + 1, 1), -1),
    ((64, 1), (1, (1 << 60) + 1), -2),
    ((1, 1, 64), ((1 << 60) + 1, 1 << 31, 1), -1),
)
INPUT_CASES = (
    "normal",
    "zero",
    "negative_zero",
    "subnormal",
    "nan",
    "minimum_scale",
    "maximum_finite",
    "positive_infinity",
    "negative_infinity",
    "scale_boundary",
)
FP32_INPUT_CASES = (
    "minimum_scale",
    "maximum_finite",
    "positive_infinity",
    "negative_infinity",
    "scale_boundary",
)


@cuda_only
@cuda_sm89_or_newer
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("shape", SHAPES + NARROW_SHAPES + LARGE_BATCH_SQUARE32_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("input_case", INPUT_CASES)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
def test_quantize_mxfp8_precision(
    dtype, contract_dim, block_shape, shape, layout, input_case, output_layout
):
    if shape in NARROW_SHAPES and input_case != "normal":
        pytest.skip("narrow shapes cover indexing across partial scale groups")
    if shape in CONTIGUOUS_B32_SHAPES and block_shape != (1, 32):
        pytest.skip("aligned B32 coverage targets 1D block-32 quantization")
    if shape in ALIGNED_SQUARE32_SHAPES and block_shape != (32, 32):
        pytest.skip("aligned square32 coverage targets 2D block-32 quantization")
    if shape in LARGE_BATCH_SQUARE32_SHAPES and (
        dtype is not torch.bfloat16
        or block_shape != (32, 32)
        or layout != "dense"
        or input_case != "normal"
        or output_layout != "row_major"
    ):
        pytest.skip("large-batch square32 coverage uses the smallest API grid")
    if shape in TAIL_SHAPES and layout == "offset":
        pytest.skip("offset coverage uses aligned B32 shapes")
    if (
        input_case != "normal"
        and block_shape[1] != 32
        and (dtype is not torch.float32 or shape != (130, 259) or layout != "dense")
    ):
        pytest.skip("non-B32 boundary coverage uses dense FP32 rank-2 inputs")
    if input_case in FP32_INPUT_CASES and dtype is not torch.float32:
        pytest.skip("FP32 scale-boundary coverage")
    source_values = torch.linspace(
        -448,
        448,
        prod(shape) + (layout == "offset"),
        device="cuda",
        dtype=dtype,
    )
    source = (
        source_values[1:].reshape(shape)
        if layout == "offset"
        else source_values.reshape(shape)
    )
    if layout == "strided":
        source = source.repeat_interleave(2, dim=-1)[..., ::2]
    elif layout == "transposed":
        source = source.mT
    if layout == "offset":
        assert source.is_contiguous()
    if input_case != "normal":
        source.zero_()
        values = source.movedim(contract_dim, -1)
        qmax = torch.finfo(torch.float8_e4m3fn).max
        if input_case in ("zero", "negative_zero", "subnormal", "nan"):
            values[..., 0, 0] = qmax
            values[..., 0, 1] = {
                "zero": 0.0,
                "negative_zero": -0.0,
                "subnormal": 2.0**-10,
                "nan": float("nan"),
            }[input_case]
        elif input_case == "minimum_scale":
            values[..., 0, 0] = qmax * 2.0**-136
        elif input_case == "maximum_finite":
            values[..., 0, 0] = torch.finfo(torch.float32).max
        elif input_case == "positive_infinity":
            values[..., 0, 0] = float("inf")
        elif input_case == "negative_infinity":
            values[..., 0, 0] = float("-inf")
        else:
            boundary = values.new_tensor(qmax * 2.0**-10)
            values[..., 0, 0] = torch.nextafter(
                boundary, values.new_tensor(float("-inf"))
            )
            values[..., 0, 1] = boundary
            values[..., 0, block_shape[1]] = torch.nextafter(
                boundary, values.new_tensor(float("inf"))
            )

    actual_codes, actual_scales, _ = quantize_mxfp8(
        source, contract_dim, block_shape, backend="cuda", output_layout=output_layout
    )
    expected_codes, expected_scales, _ = quantize_mxfp8(
        source, contract_dim, block_shape, backend="eager"
    )

    assert torch.equal(actual_codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(
        actual_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )


@cuda_only
@cuda_sm89_or_newer
@pytest.mark.parametrize("stochastic_rounding", (False, True))
@pytest.mark.parametrize("index_case", INDEX_STRIDE_CASES)
def test_quantize_mxfp8_large_strides(stochastic_rounding, index_case):
    shape, strides, contract_dim = index_case
    values = torch.full((64,), 1.0625, device="cuda", dtype=torch.bfloat16)
    values[0] = torch.finfo(torch.float8_e4m3fn).max
    source = values.as_strided(shape, strides)
    compact = values.reshape(1, 64).mT if shape[-1] == 1 else values.reshape(shape)

    torch.manual_seed(17)
    before = torch.cuda.get_rng_state()
    actual_codes, actual_scales, _ = quantize_mxfp8(
        source,
        contract_dim,
        (1, 16),
        stochastic_rounding=stochastic_rounding,
        backend="cuda",
        output_layout="column_major",
    )
    actual_state = torch.cuda.get_rng_state()
    torch.manual_seed(17)
    expected_codes, expected_scales, _ = quantize_mxfp8(
        compact,
        contract_dim,
        (1, 16),
        stochastic_rounding=stochastic_rounding,
        backend="cuda",
        output_layout="column_major",
    )
    expected_state = torch.cuda.get_rng_state()

    assert torch.equal(actual_codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(
        actual_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )
    assert torch.equal(actual_state, expected_state)
    if not stochastic_rounding:
        assert torch.equal(actual_state, before)


@cuda_only
@cuda_sm89_or_newer
@pytest.mark.parametrize("index_case", LARGE_OFFSET_CASES)
def test_quantize_mxfp8_large_offsets(index_case):
    shape, strides = index_case
    with large_offset_source(shape, strides) as source:
        source.fill_(1.0625)
        source[..., 0] = torch.finfo(torch.float8_e4m3fn).max
        compact = source.clone()
        before = torch.cuda.get_rng_state()
        actual_codes, actual_scales, _ = quantize_mxfp8(
            source, -1, (1, 16), backend="cuda", output_layout="column_major"
        )
        assert torch.equal(torch.cuda.get_rng_state(), before)
        expected_codes, expected_scales, _ = quantize_mxfp8(
            compact, -1, (1, 16), backend="cuda", output_layout="column_major"
        )
        assert torch.equal(
            actual_codes.view(torch.uint8), expected_codes.view(torch.uint8)
        )
        assert torch.equal(
            actual_scales.view(torch.uint8), expected_scales.view(torch.uint8)
        )


@cuda_only
@cuda_sm89_or_newer
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("shape", SHAPES + LARGE_BATCH_SQUARE32_SHAPES)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
def test_quantize_mxfp8_stochastic_rounding(
    dtype, block_shape, contract_dim, shape, output_layout
):
    if shape in CONTIGUOUS_B32_SHAPES and (
        block_shape != (1, 32) or contract_dim != -1
    ):
        pytest.skip("contiguous B32 coverage targets the K-major API contract")
    if shape in LARGE_BATCH_SQUARE32_SHAPES and (
        dtype is not torch.bfloat16
        or block_shape != (32, 32)
        or output_layout != "row_major"
    ):
        pytest.skip("large-batch square32 coverage uses the smallest stochastic grid")
    if shape in TAIL_SHAPES and dtype is not torch.float32:
        pytest.skip("tail stochastic-rounding coverage uses FP32")
    qmax = torch.finfo(torch.float8_e4m3fn).max
    lower, upper = 1.0, 1.125
    source = torch.full(shape, (lower + upper) / 2, device="cuda", dtype=dtype)
    source[..., :: block_shape[1]] = qmax
    if contract_dim == -2:
        source = source.mT

    torch.manual_seed(0)
    before_rne = torch.cuda.get_rng_state()
    rne_codes, rne_scales, _ = quantize_mxfp8(
        source,
        contract_dim,
        block_shape,
        stochastic_rounding=False,
        backend="cuda",
        output_layout=output_layout,
    )
    assert torch.equal(torch.cuda.get_rng_state(), before_rne)

    torch.manual_seed(1)
    before_sr = torch.cuda.get_rng_state()
    first_codes, first_scales, _ = quantize_mxfp8(
        source,
        contract_dim,
        block_shape,
        stochastic_rounding=True,
        backend="cuda",
        output_layout=output_layout,
    )
    assert not torch.equal(torch.cuda.get_rng_state(), before_sr)
    second_codes, _, _ = quantize_mxfp8(
        source,
        contract_dim,
        block_shape,
        stochastic_rounding=True,
        backend="cuda",
        output_layout=output_layout,
    )
    torch.manual_seed(1)
    repeated_codes, repeated_scales, _ = quantize_mxfp8(
        source,
        contract_dim,
        block_shape,
        stochastic_rounding=True,
        backend="cuda",
        output_layout=output_layout,
    )

    assert not torch.equal(
        first_codes.view(torch.uint8), second_codes.view(torch.uint8)
    )
    assert torch.equal(first_codes.view(torch.uint8), repeated_codes.view(torch.uint8))
    assert torch.equal(
        first_scales.view(torch.uint8), repeated_scales.view(torch.uint8)
    )
    assert torch.equal(rne_scales.view(torch.uint8), first_scales.view(torch.uint8))
    rounded = first_codes[source != qmax].float()
    assert torch.all((rounded == lower) | (rounded == upper))
    sample_count = rounded.numel()
    # 5.1 times the worst-case binomial standard error.
    atol = (upper - lower) * 5.1 / (2 * sample_count**0.5)
    torch.testing.assert_close(
        rounded.mean(), rounded.new_tensor((lower + upper) / 2), rtol=0, atol=atol
    )
    assert rne_codes.shape == first_codes.shape


@cuda_only
@cuda_sm89_or_newer
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("shape", COMPILE_SHAPES)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("layout", ("dense", "transposed"))
def test_quantize_mxfp8_compile(
    block_shape, contract_dim, shape, stochastic_rounding, output_layout, layout
):
    if shape == (33, 64) and (block_shape != (1, 32) or contract_dim != -1):
        pytest.skip("aligned B32 compilation covers the K-major API contract")
    source = torch.linspace(
        -448, 448, prod(shape), device="cuda", dtype=torch.bfloat16
    ).reshape(shape)
    if layout == "transposed":
        source = source.mT
    torch._dynamo.reset()
    quantize = torch.compile(
        lambda values: quantize_mxfp8(
            values,
            contract_dim,
            block_shape,
            stochastic_rounding=stochastic_rounding,
            backend="cuda",
            output_layout=output_layout,
        ),
        fullgraph=True,
    )

    torch.manual_seed(0)
    actual_codes, actual_scales, _ = quantize(source)
    torch.manual_seed(0)
    expected_codes, expected_scales, _ = quantize_mxfp8(
        source,
        contract_dim,
        block_shape,
        stochastic_rounding=stochastic_rounding,
        backend="cuda",
        output_layout=output_layout,
    )

    assert torch.equal(actual_codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(
        actual_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )


@cuda_only
@cuda_sm89_or_newer
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("shape", CUDA_GRAPH_SHAPES)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
def test_quantize_mxfp8_cuda_graph_stochastic_rounding(
    block_shape, shape, output_layout
):
    source = torch.full(shape, 1.0625, device="cuda")
    source[..., :: block_shape[1]] = 448
    graph = torch.cuda.CUDAGraph()

    torch.manual_seed(1)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        codes, scales, _ = quantize_mxfp8(
            source,
            block_shape=block_shape,
            stochastic_rounding=True,
            backend="cuda",
            output_layout=output_layout,
        )
    graph.replay()
    first_codes = codes.clone()
    first_scales = scales.clone()
    graph.replay()
    second_codes = codes.clone()
    second_scales = scales.clone()

    assert not torch.equal(
        first_codes.view(torch.uint8), second_codes.view(torch.uint8)
    )
    assert torch.equal(first_scales.view(torch.uint8), second_scales.view(torch.uint8))


@cuda_only
@cuda_sm89_or_newer
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_shape", ((1, 16), (32, 32)))
@pytest.mark.parametrize("stochastic_rounding", (False, True))
@pytest.mark.parametrize("shape", STAT_SHAPES)
def test_quantize_mxfp8_quantization_stats(
    output_layout, contract_dim, dtype, block_shape, stochastic_rounding, shape
):
    generator = torch.Generator(device="cuda").manual_seed(17)
    source = torch.randn(shape, device="cuda", dtype=dtype, generator=generator).mT
    if contract_dim == -1:
        source = source.mT
    if source.numel():
        source[0, 0] = 0.0
        source[0, 1] = -0.0
        source[1, 1] = source.new_tensor(1e-4)
    source.requires_grad_()
    torch.manual_seed(7)
    before = torch.cuda.get_rng_state()
    off_codes, off_scales, _ = quantize_mxfp8(
        source,
        contract_dim,
        block_shape,
        stochastic_rounding=stochastic_rounding,
        backend="cuda",
        output_layout=output_layout,
    )
    off_state = torch.cuda.get_rng_state()
    torch.manual_seed(7)
    on_codes, on_scales, _, stats = quantize_mxfp8(
        source,
        contract_dim,
        block_shape,
        stochastic_rounding=stochastic_rounding,
        backend="cuda",
        output_layout=output_layout,
        return_quantization_stats=True,
    )
    assert torch.equal(torch.cuda.get_rng_state(), off_state)
    if not stochastic_rounding:
        assert torch.equal(before, off_state)
    decoded = dequantize_dense(
        on_codes, on_scales, contract_dim, block_shape, None, backend="eager"
    )
    source_float = source.detach().float()
    nonzero = source_float != 0
    expected = torch.stack(
        (
            source_float.square().sum(),
            (source_float - decoded).square().sum(),
            (nonzero & (decoded == 0)).sum(dtype=torch.float32),
            source_float.new_tensor(source_float.numel()),
            nonzero.sum(dtype=torch.float32),
        )
    )

    assert torch.equal(on_codes.view(torch.uint8), off_codes.view(torch.uint8))
    assert torch.equal(on_scales.view(torch.uint8), off_scales.view(torch.uint8))
    assert stats is not None
    assert not stats.requires_grad
    assert torch.equal(stats[2:], expected[2:])
    energy_scale = expected[:2].clamp_min(1e-30)
    torch.testing.assert_close(
        stats[:2] / energy_scale, expected[:2] / energy_scale, rtol=0, atol=2.3e-6
    )


@cuda_only
@cuda_sm89_or_newer
@pytest.mark.parametrize("stochastic_rounding", (False, True))
@pytest.mark.parametrize("wide_stride", (False, True))
def test_quantize_mxfp8_quantization_stats_compile(stochastic_rounding, wide_stride):
    source = torch.randn((17, 35), device="cuda", dtype=torch.bfloat16)
    if wide_stride:
        values = torch.randn(35, device="cuda", dtype=torch.bfloat16)
        source = values.as_strided((1, 35), ((1 << 60) + 1, 1))
    quantize = torch.compile(
        lambda values: quantize_mxfp8(
            values,
            -1,
            (1, 16),
            stochastic_rounding=stochastic_rounding,
            backend="cuda",
            output_layout="column_major",
            return_quantization_stats=True,
        ),
        fullgraph=True,
    )
    codes, scales, global_scale, stats = quantize(source)
    assert global_scale is None
    assert codes.stride(-2) == 1
    assert scales.shape == (source.shape[-2], 3)
    assert stats.shape == (5,)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        codes, scales, global_scale, stats = quantize(source)
    for _ in range(2):
        graph.replay()
        decoded = dequantize_dense(
            codes, scales, -1, (1, 16), global_scale, backend="eager"
        )
        source_float = source.float()
        expected = torch.stack(
            (source_float.square().sum(), (source_float - decoded).square().sum())
        )
        energy_scale = expected.clamp_min(1e-30)
        torch.testing.assert_close(
            stats[:2] / energy_scale, expected / energy_scale, rtol=0, atol=2.3e-6
        )
