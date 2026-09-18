"""CUDA FP8 quantization contracts."""

from math import prod, sqrt

import pytest
import torch

from src.kernel.ops import dequantize_dense, quantize_fp8
from src.metrics.quant import accumulate_quantization_sums
from tests.fast.kernel.backends.cuda._quantize_test_utils import (
    LARGE_OFFSET_CASES,
    large_offset_source,
)
from tests.fast.helper import cuda_only


OUTPUT_LAYOUTS = ("row_major", "column_major")
STAT_SHAPES = ((35, 65), (256, 512), (1057, 33), (8192, 1024))
STAT_LAYOUTS = ("dense", "transposed", "strided")
INPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_FP8_FORMATS = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
}
CONTRACT_DIMS = (-2, -1)
BLOCK_SHAPES = (
    (0, 0),
    (1, 0),
    (1, 7),
    (1, 16),
    (1, 32),
    (1, 64),
    (1, 128),
    (1, 256),
    (1, 513),
    (7, 7),
    (32, 32),
    (64, 64),
)
SHAPES = ((2, 35, 65), (2, 1057, 33), (2, 256, 512))
LAYOUTS = ("dense", "strided", "transposed", "broadcast", "offset")
INPUT_CASES = ("normal", "finite_boundaries", "scaled_boundaries", "nonfinite")
COMPILE_BLOCK_SHAPES = ((0, 0), (1, 7), (1, 128), (32, 32))
COMPILE_SHAPES = ((35, 65), (64, 256))
INDEX_STRIDE_CASES = (
    ((1, 64), (((1 << 31) - 1) // 31, 1), -1),
    ((1, 64), (((1 << 31) - 1) // 31 + 1, 1), -1),
    ((1, 64), ((1 << 31) - 1, 1), -1),
    ((1, 64), (1 << 31, 1), -1),
    ((1, 64), ((1 << 60) + 1, 1), -1),
    ((64, 1), (1, (1 << 60) + 1), -2),
    ((1, 1, 64), ((1 << 60) + 1, 1 << 31, 1), -1),
)


@cuda_only
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("input_case", INPUT_CASES)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
def test_quantize_fp8_precision(
    input_dtype,
    fp8_dtype,
    contract_dim,
    block_shape,
    shape,
    layout,
    input_case,
    output_layout,
):
    source = torch.linspace(-500, 500, prod(shape), device="cuda").reshape(shape)
    if input_case in ("finite_boundaries", "scaled_boundaries"):
        source.zero_()
        source[0, 0, 0] = torch.finfo(fp8_dtype).max
        source[0, 0, 1] = 0.0
        source[0, 0, 2] = -0.0
        source[0, 0, 3] = 2.0**-10 if fp8_dtype is torch.float8_e4m3fn else 2.0**-17
        midpoint = source.new_tensor(
            1.0625 if fp8_dtype is torch.float8_e4m3fn else 1.125
        )
        source[0, 0, 4] = midpoint
        source[0, 0, 5] = torch.nextafter(midpoint, source.new_tensor(0.0))
        source[0, 0, 6] = torch.nextafter(midpoint, source.new_tensor(torch.inf))
        source[0, 0, 7:10] = -source[0, 0, 4:7]
        if input_case == "scaled_boundaries":
            pattern = source[0, 0, :16].clone()
            source.copy_(
                pattern[torch.arange(prod(shape), device="cuda") % 16].reshape(shape)
            )
            source *= torch.logspace(-6, 0, prod(shape[:-1]), device="cuda").reshape(
                *shape[:-1], 1
            )
    elif input_case == "nonfinite":
        source[0, 0, :3] = source.new_tensor((torch.nan, torch.inf, -torch.inf))
    source = source.to(input_dtype)
    if layout == "strided":
        source = source.repeat_interleave(2, dim=-1)[..., ::2]
    elif layout == "transposed":
        source = source.mT
    elif layout == "broadcast":
        source = source[:1].expand(*shape)
    elif layout == "offset":
        source = torch.cat((source.new_zeros(1), source.flatten()))[1:].view(shape)
    original = source.clone()

    codes, scales, _, _ = quantize_fp8(
        source,
        contract_dim,
        block_shape,
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        output_layout=output_layout,
        backend="cuda",
    )
    expected_codes, expected_scales, _, _ = quantize_fp8(
        source,
        contract_dim,
        block_shape,
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        backend="eager",
    )
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0, equal_nan=True)
    assert (codes if output_layout == "row_major" else codes.mT).is_contiguous()
    if block_shape == (0, 0):
        assert 0 in scales.stride()
    else:
        assert scales.is_contiguous()
    if input_case in ("finite_boundaries", "scaled_boundaries"):
        negative_zero = (source == 0) & torch.signbit(source)
        assert torch.all(codes.view(torch.uint8)[negative_zero] == 0x80)
    assert torch.equal(
        source.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)
    )


@cuda_only
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
@pytest.mark.parametrize("index_case", INDEX_STRIDE_CASES)
def test_quantize_fp8_large_strides(fp8_dtype, stochastic_rounding, index_case):
    shape, strides, contract_dim = index_case
    values = torch.full((64,), 1.0625, device="cuda", dtype=torch.bfloat16)
    values[0] = torch.finfo(fp8_dtype).max
    source = values.as_strided(shape, strides)
    compact = values.reshape(1, 64).mT if shape[-1] == 1 else values.reshape(shape)

    torch.manual_seed(17)
    before = torch.cuda.get_rng_state()
    actual_codes, actual_scales, _, _ = quantize_fp8(
        source,
        contract_dim,
        (1, 16),
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        stochastic_rounding=stochastic_rounding,
        output_layout="column_major",
        backend="cuda",
    )
    actual_state = torch.cuda.get_rng_state()
    torch.manual_seed(17)
    expected_codes, expected_scales, _, _ = quantize_fp8(
        compact,
        contract_dim,
        (1, 16),
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        stochastic_rounding=stochastic_rounding,
        output_layout="column_major",
        backend="cuda",
    )
    expected_state = torch.cuda.get_rng_state()

    assert torch.equal(actual_codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(actual_scales, expected_scales)
    assert torch.equal(actual_state, expected_state)
    if not stochastic_rounding:
        assert torch.equal(actual_state, before)


@cuda_only
@pytest.mark.parametrize("index_case", LARGE_OFFSET_CASES)
def test_quantize_fp8_large_offsets(index_case):
    shape, strides = index_case
    with large_offset_source(shape, strides) as source:
        source.fill_(1.0625)
        source[..., 0] = torch.finfo(torch.float8_e4m3fn).max
        compact = source.clone()
        before = torch.cuda.get_rng_state()
        actual_codes, actual_scales, _, _ = quantize_fp8(
            source, -1, (32, 32), output_layout="column_major", backend="cuda"
        )
        assert torch.equal(torch.cuda.get_rng_state(), before)
        expected_codes, expected_scales, _, _ = quantize_fp8(
            compact, -1, (32, 32), output_layout="column_major", backend="cuda"
        )
        assert torch.equal(
            actual_codes.view(torch.uint8), expected_codes.view(torch.uint8)
        )
        assert torch.equal(actual_scales, expected_scales)


@cuda_only
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
def test_quantize_fp8_stochastic_rounding(fp8_dtype, block_shape):
    rows, cols = 256, 256
    source = torch.full((rows, cols), 1.0625, device="cuda")
    subnormal = 2.0**-10 if fp8_dtype is torch.float8_e4m3fn else 2.0**-17
    source[:, 1::2] = subnormal
    qmax = torch.finfo(fp8_dtype).max
    outer, contract = block_shape
    if block_shape == (0, 0):
        source[0, 0] = qmax
    elif block_shape == (1, 0):
        source[:, 0] = qmax
    elif outer == 1:
        source[:, ::contract] = qmax
    else:
        for row in range(0, rows, outer):
            for col in range(0, cols, contract):
                source[row, col] = qmax

    torch.manual_seed(0)
    before_rne = torch.cuda.get_rng_state()
    rne_codes, rne_scales, _, _ = quantize_fp8(
        source,
        -1,
        block_shape,
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        backend="cuda",
    )
    assert torch.equal(torch.cuda.get_rng_state(), before_rne)

    torch.manual_seed(1)
    before_sr = torch.cuda.get_rng_state()
    first_codes, first_scales, _, _ = quantize_fp8(
        source,
        -1,
        block_shape,
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        stochastic_rounding=True,
        backend="cuda",
    )
    assert not torch.equal(torch.cuda.get_rng_state(), before_sr)
    torch.manual_seed(1)
    second_codes, second_scales, _, _ = quantize_fp8(
        source,
        -1,
        block_shape,
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        stochastic_rounding=True,
        backend="cuda",
    )

    assert torch.equal(first_codes.view(torch.uint8), second_codes.view(torch.uint8))
    assert torch.equal(first_scales, second_scales)
    assert torch.equal(rne_scales, first_scales)
    normal_mask = source == 1.0625
    normal = first_codes.float()[normal_mask]
    normal_upper = 1.125 if fp8_dtype is torch.float8_e4m3fn else 1.25
    assert torch.all((normal == 1.0) | (normal == normal_upper))
    subnormal_mask = source == subnormal
    rounded_subnormal = first_codes.float()[subnormal_mask]
    subnormal_upper = 2.0**-9 if fp8_dtype is torch.float8_e4m3fn else 2.0**-16
    assert torch.all(
        (rounded_subnormal == 0.0) | (rounded_subnormal == subnormal_upper)
    )
    for rounded, lower, expected, upper in (
        (normal, 1.0, 1.0625, normal_upper),
        (rounded_subnormal, 0.0, subnormal, subnormal_upper),
    ):
        probability = (expected - lower) / (upper - lower)
        # 4.7 standard errors bound the binomial rounding mean.
        atol = (
            4.7
            * (upper - lower)
            * sqrt(probability * (1 - probability) / rounded.numel())
        )
        torch.testing.assert_close(
            rounded.mean(), rounded.new_tensor(expected), rtol=0, atol=atol
        )
    assert rne_codes.shape == first_codes.shape


def _capture_graph(quantize, source):
    static_source = source.clone()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        codes, scales, _, stats = quantize(static_source)
    assert stats is False
    return graph, static_source, codes, scales


@cuda_only
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize("shape", COMPILE_SHAPES)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
def test_quantize_fp8_compile_replay(
    fp8_dtype, contract_dim, block_shape, shape, stochastic_rounding
):
    source = torch.linspace(-448, 448, prod(shape), device="cuda", dtype=torch.bfloat16)
    source = source.reshape(shape)
    torch._dynamo.reset()
    quantize = torch.compile(
        lambda values: quantize_fp8(
            values,
            contract_dim,
            block_shape,
            fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
            stochastic_rounding=stochastic_rounding,
            backend="cuda",
        ),
        fullgraph=True,
    )

    expected_codes, expected_scales, _, _ = quantize_fp8(
        source,
        contract_dim,
        block_shape,
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        backend="eager",
    )
    quantize(source)
    original = source.clone()
    torch.manual_seed(17)
    graph, static_source, graph_codes, graph_scales = _capture_graph(quantize, source)
    graph.replay()
    torch.cuda.synchronize()

    first_codes = graph_codes.clone()
    first_scales = graph_scales.clone()
    assert torch.equal(static_source, original)
    assert torch.equal(source, original)
    if not stochastic_rounding:
        assert torch.equal(
            first_codes.view(torch.uint8), expected_codes.view(torch.uint8)
        )
        assert torch.equal(first_scales, expected_scales)
        return

    graph.replay()
    torch.cuda.synchronize()
    second_codes = graph_codes.clone()
    second_scales = graph_scales.clone()
    assert not torch.equal(
        first_codes.view(torch.uint8), second_codes.view(torch.uint8)
    )
    assert torch.equal(first_scales, second_scales)

    torch.manual_seed(17)
    reset_graph, reset_source, reset_codes, reset_scales = _capture_graph(
        quantize, source
    )
    reset_graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(first_codes.view(torch.uint8), reset_codes.view(torch.uint8))
    assert torch.equal(first_scales, reset_scales)
    assert torch.equal(reset_source, original)


@cuda_only
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("shape", STAT_SHAPES)
@pytest.mark.parametrize("layout", STAT_LAYOUTS)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
def test_quantize_fp8_statistics_precision(
    input_dtype,
    fp8_dtype,
    contract_dim,
    block_shape,
    shape,
    layout,
    output_layout,
    stochastic_rounding,
):
    generator = torch.Generator(device="cuda").manual_seed(17)
    source = torch.randn(shape, device="cuda", generator=generator).to(input_dtype)
    source[::7, ::3] = 0
    source[1::7, 1::3] *= 1e-4
    if layout == "transposed":
        source = source.mT
    elif layout == "strided":
        source = source.repeat_interleave(2, dim=-1)[..., ::2]
    source.requires_grad_(True)
    torch.manual_seed(123)
    ordinary = quantize_fp8(
        source,
        contract_dim,
        block_shape,
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        stochastic_rounding=stochastic_rounding,
        output_layout=output_layout,
        backend="cuda",
    )
    ordinary_rng = torch.cuda.get_rng_state()
    torch.manual_seed(123)
    codes, scales, global_scale, stats = quantize_fp8(
        source,
        contract_dim,
        block_shape,
        fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
        stochastic_rounding=stochastic_rounding,
        output_layout=output_layout,
        return_quantization_stats=True,
        backend="cuda",
    )
    assert torch.equal(torch.cuda.get_rng_state(), ordinary_rng)
    assert torch.equal(
        codes.contiguous().view(torch.uint8), ordinary[0].contiguous().view(torch.uint8)
    )
    assert torch.equal(scales, ordinary[1])
    assert global_scale is None
    assert stats.shape == (5,) and stats.dtype is torch.float32
    assert stats.device == source.device and not stats.requires_grad
    reconstructed = dequantize_dense(codes, scales, contract_dim, block_shape)
    expected = torch.cat(
        accumulate_quantization_sums(
            source.detach(),
            codes,
            reconstructed,
            contract_dim=contract_dim,
        )
    )
    torch.testing.assert_close(stats[2:], expected[2:], atol=0, rtol=0)
    energy_scale = expected[:2].clamp_min(1e-30)
    # Maximum normalized energy error was 1.82e-4 across the 22,080-case grid.
    torch.testing.assert_close(
        stats[:2] / energy_scale, expected[:2] / energy_scale, atol=7.3e-4, rtol=0
    )


@cuda_only
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
@pytest.mark.parametrize("wide_stride", (False, True))
def test_quantize_fp8_statistics_compile(
    contract_dim, output_layout, stochastic_rounding, wide_stride
):
    source = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    if wide_stride:
        values = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        source = (
            values.as_strided((1, 256), ((1 << 60) + 1, 1))
            if contract_dim == -1
            else values.as_strided((256, 1), (1, (1 << 60) + 1))
        )
    fp8_dtype = torch.float8_e4m3fn
    block_shape = (1, 128)

    def quantize(source):
        return quantize_fp8(
            source,
            contract_dim,
            block_shape,
            fmt=_FP8_FORMATS.get(fp8_dtype, "invalid"),
            stochastic_rounding=stochastic_rounding,
            output_layout=output_layout,
            return_quantization_stats=True,
            backend="cuda",
        )

    torch.compiler.reset()
    compiled = torch.compile(quantize, fullgraph=True)
    compiled(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        codes, scales, _, stats = compiled(source)
    for _ in range(2):
        graph.replay()
        reconstructed = dequantize_dense(codes, scales, contract_dim, block_shape)
        expected = torch.cat(
            accumulate_quantization_sums(
                source,
                codes,
                reconstructed,
                contract_dim=contract_dim,
            )
        )
        torch.testing.assert_close(stats[2:], expected[2:], atol=0, rtol=0)
        energy_scale = expected[:2].clamp_min(1e-30)
        torch.testing.assert_close(
            stats[:2] / energy_scale, expected[:2] / energy_scale, atol=2.3e-6, rtol=0
        )
        assert (codes if output_layout == "row_major" else codes.mT).is_contiguous()
