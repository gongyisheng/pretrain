"""Native NVFP4 quantization parity tests."""

import pytest
import torch

from src.kernel.ops import dequantize_dense, quantize_nvfp4, unpack_e2m1
from src.quant.quantize import dequantize_operand
from tests.fast.helper import cuda_only, cuda_sm100_or_newer


DTYPES = (torch.float32, torch.float16, torch.bfloat16)
CONTRACT_DIMS = (-2, -1)
BLOCK_SHAPES = ((1, 16), (16, 16))
GLOBAL_SCALES = (False, True)
QMAX_VALUES = (6.0, 4.0)
STOCHASTIC_CASES = ("rounding", "nonfinite", "large")
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
        enable_global_scale,
        backend="eager",
        qmax=qmax,
    )
    actual = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        enable_global_scale,
        backend="cuda",
        qmax=qmax,
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
        enable_global_scale,
        stochastic_rounding=True,
        backend="eager",
        qmax=qmax,
    )
    actual = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        enable_global_scale,
        stochastic_rounding=True,
        backend="cuda",
        qmax=qmax,
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
        enable_global_scale,
        backend="eager",
        qmax=qmax,
    )

    def quantize(x):
        return quantize_nvfp4(
            x, -1, block_shape, enable_global_scale, backend="cuda", qmax=qmax
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
            x, contract_dim, block_shape, enable_global_scale, backend="cuda", qmax=qmax
        )

    compiled = torch.compile(quantize, fullgraph=True, dynamic=True)
    for index, shape in enumerate(DYNAMIC_SHAPES[rank][contract_dim]):
        source = _make_source(shape, torch.bfloat16, "strided")
        expected = quantize_nvfp4(
            source,
            contract_dim,
            block_shape,
            enable_global_scale,
            backend="eager",
            qmax=qmax,
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
def test_quantize_nvfp4_stochastic_rounding_rng(block_shape, enable_global_scale, qmax):
    source = torch.full((4096, 16), 0.3, device="cuda")
    source[:, -1] = qmax
    torch.manual_seed(0)
    initial_state = torch.cuda.get_rng_state()
    quantize_nvfp4(
        source, -1, block_shape, enable_global_scale, backend="cuda", qmax=qmax
    )
    assert torch.equal(torch.cuda.get_rng_state(), initial_state)

    def quantize(x):
        return quantize_nvfp4(
            x,
            -1,
            block_shape,
            enable_global_scale,
            stochastic_rounding=True,
            backend="cuda",
            qmax=qmax,
        )

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
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = quantize(source)
    graph.replay()
    first_replay = captured[0].clone()
    graph.replay()
    second_replay = captured[0].clone()
    assert not torch.equal(first_replay, second_replay)


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("output_layout", ("row_major", "column_major"))
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
def test_quantize_nvfp4_quantization_stats(
    output_layout,
    contract_dim,
    dtype,
    block_shape,
    enable_global_scale,
    qmax,
    stochastic_rounding,
):
    source = _make_source((17, 64), dtype, "strided")
    if contract_dim == -2:
        source = source.mT
    source[0, 0] = 0.0
    source[0, 1] = -0.0
    source[1, 1] = source.new_tensor(1e-4)
    source.requires_grad_()
    torch.manual_seed(7)
    before = torch.cuda.get_rng_state()
    off_codes, off_scales, off_global = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        enable_global_scale,
        stochastic_rounding=stochastic_rounding,
        backend="cuda",
        qmax=qmax,
        output_layout=output_layout,
    )
    off_state = torch.cuda.get_rng_state()
    torch.manual_seed(7)
    on_codes, on_scales, on_global, stats = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        enable_global_scale,
        stochastic_rounding=stochastic_rounding,
        backend="cuda",
        qmax=qmax,
        output_layout=output_layout,
        return_quantization_stats=True,
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
    torch.testing.assert_close(
        stats[:2] / energy_scale, expected[:2] / energy_scale, rtol=0, atol=2.3e-6
    )


@cuda_only
@cuda_sm100_or_newer
@pytest.mark.parametrize("qmax", QMAX_VALUES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("enable_global_scale", GLOBAL_SCALES)
@pytest.mark.parametrize("output_layout", ("row_major", "column_major"))
@pytest.mark.parametrize("stochastic_rounding", (False, True))
def test_quantize_nvfp4_quantization_stats_compile(
    qmax, contract_dim, enable_global_scale, output_layout, stochastic_rounding
):
    torch.compiler.reset()
    source = _make_source((17, 64), torch.bfloat16, "strided")
    if contract_dim == -2:
        source = source.mT
    block_shape = (1, 16)
    quantize = torch.compile(
        lambda values: quantize_nvfp4(
            values,
            contract_dim,
            block_shape,
            enable_global_scale=enable_global_scale,
            stochastic_rounding=stochastic_rounding,
            backend="cuda",
            qmax=qmax,
            output_layout=output_layout,
            return_quantization_stats=True,
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
