"""Native NVFP4 quantization parity tests."""

import pytest
import torch

from src.kernel.ops import dequantize_dense, quantize_nvfp4, unpack_e2m1
from src.kernel.utils import to_swizzle_32_4_4
from src.quant.quantize import dequantize_operand
from tests.fast.kernel.backends.cuda._quantize_test_utils import (
    LARGE_OFFSET_CASES,
    large_offset_source,
)
from tests.fast.helper import cuda_only, cuda_sm100_or_newer


DTYPES = (torch.float32, torch.float16, torch.bfloat16)
CONTRACT_DIMS = (-2, -1)
BLOCK_SHAPES = ((1, 16), (16, 16))
GLOBAL_SCALES = (False, True)
QMAX_VALUES = (6.0, 4.0)
SWIZZLE_SHAPES = (
    (1, 16),
    (16, 16),
    (17, 80),
    (128, 128),
    (144, 64),
    (128, 80),
    (144, 80),
)
SWIZZLE_INPUT_LAYOUTS = ("dense", "strided")
STAT_SHAPES = ((17, 64), (8192, 1024))
OUTPUT_LAYOUTS = ("row_major", "column_major")
SCALE_LAYOUTS = ("row_major", "swizzled_32_4_4")
RETURN_QUANTIZATION_STATS = (False, True)
STOCHASTIC_CASES = ("rounding", "nonfinite", "large")
STOCHASTIC_ROUNDING = (False, True)
LAYOUTS = ("dense", "strided")
SHAPES = ((32, 64), (17, 64), (64, 17), (2, 32, 64), (2, 65, 64), (2, 64, 65))
DYNAMIC_RANKS = (2, 3)
DYNAMIC_SHAPES = {
    2: {
        -2: ((64, 17), (80, 33)),
        -1: ((17, 64), (33, 80)),
    },
    3: {
        -2: ((2, 64, 17), (3, 80, 33)),
        -1: ((2, 17, 64), (3, 33, 80)),
    },
}
SPECIAL_VALUES = (
    "boundary",
    "scale_boundaries",
    "nonfinite",
    "tiny",
    "zero",
    "subnormal",
    "large",
)
INDEX_STRIDE_CASES = (
    ((1, 64), (((1 << 31) - 1) // 31, 1), -1),
    ((1, 64), (((1 << 31) - 1) // 31 + 1, 1), -1),
    ((1, 64), ((1 << 31) - 1, 1), -1),
    ((1, 64), (1 << 31, 1), -1),
    ((1, 64), ((1 << 60) + 1, 1), -1),
    ((64, 1), (1, (1 << 60) + 1), -2),
    ((1, 1, 64), ((1 << 60) + 1, 1 << 31, 1), -1),
)


def _make_source(shape, dtype, layout):
    generator = torch.Generator(device="cuda").manual_seed(0)
    source = torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
    source = source.to(dtype)
    if layout == "strided":
        source = source.mT.contiguous().mT
    return source


def _scale_config(block_shape, enable_global_scale):
    return {
        "granularity": "blockwise",
        "block_shape": block_shape,
        "scale_dtype": torch.float8_e4m3fn,
        "enable_global_scale": enable_global_scale,
    }


def _statistics_oracle(source, codes, scales, global_scale, contract_dim, block_shape):
    decoded = dequantize_dense(
        codes, scales, contract_dim, block_shape, global_scale, backend="eager"
    )
    source_float = source.detach().float()
    nonzero = source_float != 0
    return torch.stack(
        (
            source_float.square().sum(),
            (source_float - decoded).square().sum(),
            (nonzero & (decoded == 0)).sum(dtype=torch.float32),
            source_float.new_tensor(source_float.numel()),
            nonzero.sum(dtype=torch.float32),
        )
    )


def _assert_rne_matches_eager(
    source, contract_dim, block_shape, enable_global_scale, qmax
):
    expected = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        backend="eager",
    )
    actual = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        backend="cuda",
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(
        actual[1].contiguous().view(torch.uint8),
        expected[1].contiguous().view(torch.uint8),
    )
    if enable_global_scale:
        assert actual[2] is not None
        assert torch.equal(
            actual[2].contiguous().view(torch.uint8),
            expected[2].contiguous().view(torch.uint8),
        )
    else:
        assert actual[2] is None
    assert actual[3] is False
    assert expected[3] is False


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", SHAPES)
def test_quantize_nvfp4_precision(
    dtype, contract_dim, block_shape, enable_global_scale, layout, shape, qmax
):
    if shape[contract_dim] % 16:
        pytest.skip("NVFP4 requires a contraction extent divisible by 16")
    source = _make_source(shape, dtype, layout)
    original = source.clone()
    _assert_rne_matches_eager(
        source, contract_dim, block_shape, enable_global_scale, qmax
    )
    assert torch.equal(source, original)


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SWIZZLE_SHAPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("input_layout", SWIZZLE_INPUT_LAYOUTS)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("stochastic_rounding", STOCHASTIC_ROUNDING)
@pytest.mark.parametrize("return_quantization_stats", RETURN_QUANTIZATION_STATS)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
def test_quantize_nvfp4_scale_layout(
    dtype,
    shape,
    contract_dim,
    block_shape,
    input_layout,
    output_layout,
    stochastic_rounding,
    return_quantization_stats,
    enable_global_scale,
    qmax,
):
    if shape[contract_dim] % 16:
        pytest.skip("NVFP4 requires a contraction extent divisible by 16")
    source = _make_source(shape, dtype, input_layout)
    fmt = "fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1"
    torch.manual_seed(17)
    codes, scales, global_scale, stats = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        fmt=fmt,
        enable_global_scale=enable_global_scale,
        stochastic_rounding=stochastic_rounding,
        output_layout=output_layout,
        return_quantization_stats=return_quantization_stats,
        backend="cuda",
    )
    state = torch.cuda.get_rng_state()
    torch.manual_seed(17)
    packed_codes, packed_scales, packed_global_scale, packed_stats = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        fmt=fmt,
        enable_global_scale=enable_global_scale,
        stochastic_rounding=stochastic_rounding,
        output_layout=output_layout,
        return_quantization_stats=return_quantization_stats,
        scale_layout="swizzled_32_4_4",
        backend="cuda",
    )
    packed_state = torch.cuda.get_rng_state()

    expected_scales = to_swizzle_32_4_4(scales if contract_dim == -1 else scales.t())
    assert torch.equal(packed_codes, codes)
    assert torch.equal(
        packed_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )
    if enable_global_scale:
        assert global_scale is not None
        assert torch.equal(packed_global_scale, global_scale)
    else:
        assert global_scale is None
        assert packed_global_scale is None
    assert torch.equal(packed_state, state)
    if return_quantization_stats:
        assert torch.equal(packed_stats[2:], stats[2:])
        energy_scale = stats[:2].clamp_min(1e-30)
        torch.testing.assert_close(
            packed_stats[:2] / energy_scale,
            stats[:2] / energy_scale,
            rtol=0,
            atol=2.3e-6,
        )
    else:
        assert stats is False
        assert packed_stats is False


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("stochastic_rounding", (False, True))
@pytest.mark.parametrize("index_case", INDEX_STRIDE_CASES)
def test_quantize_nvfp4_large_strides(stochastic_rounding, index_case):
    shape, strides, contract_dim = index_case
    values = torch.full((64,), 1.25, device="cuda", dtype=torch.bfloat16)
    values[0] = 6.0
    source = values.as_strided(shape, strides)
    compact = values.reshape(1, 64).mT if shape[-1] == 1 else values.reshape(shape)

    torch.manual_seed(17)
    before = torch.cuda.get_rng_state()
    actual_codes, actual_scales, actual_global_scale, _ = quantize_nvfp4(
        source,
        contract_dim,
        (1, 16),
        stochastic_rounding=stochastic_rounding,
        output_layout="column_major",
        backend="cuda",
    )
    actual_state = torch.cuda.get_rng_state()
    torch.manual_seed(17)
    expected_codes, expected_scales, expected_global_scale, _ = quantize_nvfp4(
        compact,
        contract_dim,
        (1, 16),
        stochastic_rounding=stochastic_rounding,
        output_layout="column_major",
        backend="cuda",
    )
    expected_state = torch.cuda.get_rng_state()

    assert torch.equal(actual_codes, expected_codes)
    assert torch.equal(
        actual_scales.contiguous().view(torch.uint8),
        expected_scales.contiguous().view(torch.uint8),
    )
    assert torch.equal(actual_global_scale, expected_global_scale)
    assert torch.equal(actual_state, expected_state)
    if not stochastic_rounding:
        assert torch.equal(actual_state, before)


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("index_case", LARGE_OFFSET_CASES)
def test_quantize_nvfp4_large_offsets(index_case):
    shape, strides = index_case
    with large_offset_source(shape, strides) as source:
        source.fill_(1.25)
        source[..., 0] = 6.0
        compact = source.clone()
        before = torch.cuda.get_rng_state()
        actual_codes, actual_scales, actual_global_scale, _ = quantize_nvfp4(
            source, -1, (1, 16), output_layout="column_major", backend="cuda"
        )
        assert torch.equal(torch.cuda.get_rng_state(), before)
        expected_codes, expected_scales, expected_global_scale, _ = quantize_nvfp4(
            compact, -1, (1, 16), output_layout="column_major", backend="cuda"
        )
        assert torch.equal(actual_codes, expected_codes)
        assert torch.equal(
            actual_scales.contiguous().view(torch.uint8),
            expected_scales.contiguous().view(torch.uint8),
        )
        assert torch.equal(actual_global_scale, expected_global_scale)


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("special_value", SPECIAL_VALUES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
def test_quantize_nvfp4_special_values_precision(
    special_value, contract_dim, block_shape, enable_global_scale, qmax
):
    values = torch.tensor(
        (
            0.0,
            -0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
        ),
        device="cuda",
    )
    source = values.repeat(17, 4)
    if special_value == "scale_boundaries":
        scales = (
            torch.arange(1, 127, dtype=torch.uint8, device="cuda")
            .view(torch.float8_e4m3fn)
            .float()
        )
        midpoints = scales[:, None] * source.new_tensor(
            (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
        )
        neighbors = torch.stack(
            (
                torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf)),
                midpoints,
                torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf)),
            )
        )
        probes = torch.cat(
            (
                neighbors,
                -neighbors,
                (qmax * scales).expand(3, -1)[..., None],
                source.new_zeros(3, scales.numel(), 1),
            ),
            dim=-1,
        ).reshape(-1, 16)
        source = probes[None] * source.new_tensor((1.0, 1.7, 1e-8))[:, None, None]
    elif special_value == "nonfinite":
        source[0, :4] = source.new_tensor((torch.nan, torch.inf, -torch.inf, -0.0))
    elif special_value == "tiny":
        source.fill_(2.0**-20)
    elif special_value == "zero":
        source.zero_()
        source[:, 1] = -0.0
    elif special_value == "subnormal":
        source.fill_(2.0**-140)
        source[:, 1] = -(2.0**-140)
    elif special_value == "large":
        source *= 1e35
    if contract_dim == -2:
        source = source.mT
    _assert_rne_matches_eager(
        source, contract_dim, block_shape, enable_global_scale, qmax
    )


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
@pytest.mark.parametrize("case", STOCHASTIC_CASES)
def test_quantize_nvfp4_stochastic_rounding_precision(
    contract_dim, block_shape, enable_global_scale, qmax, case
):
    samples = 4096
    source = torch.full((samples, 16), 0.3, device="cuda")
    source[:, -1] = qmax
    if case == "nonfinite":
        source.zero_()
        source[0, :3] = source.new_tensor((torch.nan, torch.inf, -torch.inf))
        source[16, :2] = source.new_tensor((torch.inf, -torch.inf))
    elif case == "large":
        source.fill_(qmax * 1e35)
        source[:, 0] *= -1
    if contract_dim == -2:
        source = source.mT
    expected = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        stochastic_rounding=True,
        backend="eager",
    )
    actual = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        stochastic_rounding=True,
        backend="cuda",
    )

    assert torch.equal(
        actual[1].contiguous().view(torch.uint8),
        expected[1].contiguous().view(torch.uint8),
    )
    if enable_global_scale:
        assert actual[2] is not None
        assert torch.equal(
            actual[2].contiguous().view(torch.uint8),
            expected[2].contiguous().view(torch.uint8),
        )
    else:
        assert actual[2] is None
    if qmax == 4.0:
        assert torch.all((actual[0] & 7) <= 6)
        assert torch.all(((actual[0] >> 4) & 7) <= 6)
    if case != "rounding":
        assert torch.equal(actual[0], expected[0])
        return
    dequantized = dequantize_operand(
        actual[0],
        actual[1],
        contract_dim,
        _scale_config(block_shape, enable_global_scale),
        global_scale=actual[2],
    )
    samples = dequantized[:, 0] if contract_dim == -1 else dequantized[0, :]
    assert torch.all((samples == 0.0) | (samples == 0.5))
    torch.testing.assert_close(
        samples.mean(), samples.new_tensor(0.3), rtol=0, atol=0.028
    )


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
def test_quantize_nvfp4_compiles_fullgraph(block_shape, enable_global_scale, qmax):
    source = _make_source((17, 64), torch.bfloat16, "strided")
    expected = quantize_nvfp4(
        source,
        -1,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        backend="eager",
    )

    def quantize(x):
        return quantize_nvfp4(
            x,
            -1,
            block_shape,
            fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
            enable_global_scale=enable_global_scale,
            backend="cuda",
        )

    actual = torch.compile(quantize, fullgraph=True)(source)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(
        actual[1].contiguous().view(torch.uint8),
        expected[1].contiguous().view(torch.uint8),
    )
    if enable_global_scale:
        assert torch.equal(
            actual[2].contiguous().view(torch.uint8),
            expected[2].contiguous().view(torch.uint8),
        )
    else:
        assert actual[2] is None


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("rank", DYNAMIC_RANKS)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
def test_quantize_nvfp4_compiles_dynamic_shape(
    rank, contract_dim, block_shape, enable_global_scale, qmax
):
    torch.compiler.reset()

    def quantize(x):
        return quantize_nvfp4(
            x,
            contract_dim,
            block_shape,
            fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
            enable_global_scale=enable_global_scale,
            backend="cuda",
        )

    compiled = torch.compile(quantize, fullgraph=True, dynamic=True)
    for index, shape in enumerate(DYNAMIC_SHAPES[rank][contract_dim]):
        source = _make_source(shape, torch.bfloat16, "strided")
        expected = quantize_nvfp4(
            source,
            contract_dim,
            block_shape,
            fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
            enable_global_scale=enable_global_scale,
            backend="eager",
        )
        if index == 0:
            # Resolve dispatch before checking recompilation from shape changes.
            quantize(source)
            actual = compiled(source)
        else:
            with torch.compiler.set_stance("fail_on_recompile"):
                actual = compiled(source)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(
            actual[1].contiguous().view(torch.uint8),
            expected[1].contiguous().view(torch.uint8),
        )
        if enable_global_scale:
            assert torch.equal(
                actual[2].contiguous().view(torch.uint8),
                expected[2].contiguous().view(torch.uint8),
            )
        else:
            assert actual[2] is None


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
@pytest.mark.parametrize("scale_layout", SCALE_LAYOUTS)
def test_quantize_nvfp4_stochastic_rounding_rng(
    block_shape, enable_global_scale, qmax, scale_layout
):
    source = torch.full((4096, 16), 0.3, device="cuda")
    source[:, -1] = qmax
    torch.manual_seed(0)
    initial_state = torch.cuda.get_rng_state()
    quantize_nvfp4(
        source,
        -1,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        backend="cuda",
    )
    assert torch.equal(torch.cuda.get_rng_state(), initial_state)

    def quantize(x):
        return quantize_nvfp4(
            x,
            -1,
            block_shape,
            fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
            enable_global_scale=enable_global_scale,
            stochastic_rounding=True,
            backend="cuda",
            scale_layout=scale_layout,
        )

    torch._dynamo.reset()
    compiled = torch.compile(quantize, fullgraph=True)
    compiled(source)
    torch.cuda.synchronize()
    torch.manual_seed(1)
    first = compiled(source)
    second = compiled(source)
    torch.manual_seed(1)
    repeated = compiled(source)
    assert not torch.equal(first[0], second[0])
    assert torch.equal(first[0], repeated[0])

    for _ in range(3):
        quantize(source)
    torch.cuda.synchronize()
    _, row_major_scales, _, _ = quantize_nvfp4(
        source,
        -1,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        backend="cuda",
    )
    expected_scales = (
        row_major_scales
        if scale_layout == "row_major"
        else to_swizzle_32_4_4(row_major_scales)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = quantize(source)
    graph.replay()
    first_replay = captured[0].clone()
    first_scales = captured[1].clone()
    graph.replay()
    second_replay = captured[0].clone()
    second_scales = captured[1].clone()
    assert not torch.equal(first_replay, second_replay)
    assert torch.equal(first_scales.view(torch.uint8), second_scales.view(torch.uint8))
    assert torch.equal(
        first_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )

    source[: block_shape[0], -1] = qmax / 2
    _, row_major_scales, _, _ = quantize_nvfp4(
        source,
        -1,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        backend="cuda",
    )
    expected_scales = (
        row_major_scales
        if scale_layout == "row_major"
        else to_swizzle_32_4_4(row_major_scales)
    )
    graph.replay()
    changed_scales = captured[1].clone()

    assert not torch.equal(
        first_scales.view(torch.uint8), changed_scales.view(torch.uint8)
    )
    assert torch.equal(
        changed_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("output_layout", ("row_major", "column_major"))
@pytest.mark.parametrize("input_layout", LAYOUTS)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
@pytest.mark.parametrize("shape", STAT_SHAPES)
def test_quantize_nvfp4_quantization_stats(
    output_layout,
    input_layout,
    contract_dim,
    dtype,
    block_shape,
    enable_global_scale,
    qmax,
    stochastic_rounding,
    shape,
):
    source = _make_source(shape, dtype, input_layout)
    if contract_dim == -2:
        source = source.mT
    source[0, 0] = 0.0
    source[0, 1] = -0.0
    source[1, 1] = source.new_tensor(1e-4)
    source.requires_grad_()
    torch.manual_seed(7)
    before = torch.cuda.get_rng_state()
    off_codes, off_scales, off_global, _ = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        stochastic_rounding=stochastic_rounding,
        output_layout=output_layout,
        backend="cuda",
    )
    off_state = torch.cuda.get_rng_state()
    torch.manual_seed(7)
    on_codes, on_scales, on_global, stats = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
        enable_global_scale=enable_global_scale,
        stochastic_rounding=stochastic_rounding,
        output_layout=output_layout,
        return_quantization_stats=True,
        backend="cuda",
    )
    assert torch.equal(torch.cuda.get_rng_state(), off_state)
    if not stochastic_rounding:
        assert torch.equal(before, off_state)
    expected = _statistics_oracle(
        source, on_codes, on_scales, on_global, contract_dim, block_shape
    )

    assert torch.equal(on_codes, off_codes)
    assert torch.equal(on_scales.view(torch.uint8), off_scales.view(torch.uint8))
    if on_global is None:
        assert off_global is None
    else:
        assert torch.equal(on_global, off_global)
    assert stats is not None and stats.dtype is torch.float32
    assert stats.device == source.device and not stats.requires_grad
    assert (on_codes if output_layout == "row_major" else on_codes.mT).is_contiguous()
    unpacked = unpack_e2m1(on_codes, dim=contract_dim, backend="eager")
    assert torch.signbit(unpacked[0, 1])
    assert torch.equal(stats[2:], expected[2:])
    energy_scale = expected[:2].clamp_min(1e-30)
    # Maximum normalized energy error was 3.10e-6 across the 22,080-case grid.
    torch.testing.assert_close(
        stats[:2] / energy_scale, expected[:2] / energy_scale, rtol=0, atol=1.3e-5
    )


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
@pytest.mark.parametrize("output_layout", ("row_major", "column_major"))
@pytest.mark.parametrize("stochastic_rounding", (False, True))
@pytest.mark.parametrize("wide_stride", (False, True))
def test_quantize_nvfp4_quantization_stats_compile(
    qmax,
    contract_dim,
    enable_global_scale,
    output_layout,
    stochastic_rounding,
    wide_stride,
):
    torch.compiler.reset()
    source = _make_source((17, 64), torch.bfloat16, "strided")
    if wide_stride:
        values = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        source = (
            values.as_strided((1, 64), ((1 << 60) + 1, 1))
            if contract_dim == -1
            else values.as_strided((64, 1), (1, (1 << 60) + 1))
        )
    elif contract_dim == -2:
        source = source.mT
    block_shape = (1, 16)
    quantize = torch.compile(
        lambda values: quantize_nvfp4(
            values,
            contract_dim,
            block_shape,
            fmt="fp4_e2m1_4over6" if qmax == 4.0 else "fp4_e2m1",
            enable_global_scale=enable_global_scale,
            stochastic_rounding=stochastic_rounding,
            output_layout=output_layout,
            return_quantization_stats=True,
            backend="cuda",
        ),
        fullgraph=True,
    )
    codes, scales, global_scale, stats = quantize(source)
    assert (codes if output_layout == "row_major" else codes.mT).is_contiguous()
    if enable_global_scale:
        assert global_scale is not None
    else:
        assert global_scale is None
    assert stats.shape == (5,) and stats.dtype is torch.float32

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        codes, scales, global_scale, stats = quantize(source)
    for _ in range(2):
        graph.replay()
        expected = _statistics_oracle(
            source, codes, scales, global_scale, contract_dim, block_shape
        )
        energy_scale = expected[:2].clamp_min(1e-30)
        torch.testing.assert_close(
            stats[:2] / energy_scale,
            expected[:2] / energy_scale,
            rtol=0,
            atol=2.3e-6,
        )
        assert torch.equal(stats[2:], expected[2:])
