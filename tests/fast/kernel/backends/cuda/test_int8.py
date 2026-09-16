"""Dense CUDA integer quantization contracts."""

from math import prod, sqrt

import pytest
import torch

from src.kernel.ops import dequantize_dense, quantize_int8
from src.metrics.quant import accumulate_quantization_sums
from tests.fast.helper import cuda_only


OUTPUT_LAYOUTS = ("row_major", "column_major")
STAT_SHAPES = ((35, 65), (256, 512), (1057, 33))
STAT_LAYOUTS = ("dense", "transposed", "strided")
INPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
BITS = (4, 5, 6, 7, 8)
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
    (7, 7),
    (32, 32),
)
SHAPES = ((35, 65), (2, 1057, 33), (2, 64, 256))
LAYOUTS = ("dense", "strided", "transposed", "broadcast", "offset")
INPUT_CASES = ("normal", "boundaries", "nonfinite", "tiny")
COMPILE_BLOCK_SHAPES = ((0, 0), (1, 0), (1, 32), (32, 32))


@cuda_only
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("input_case", INPUT_CASES)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
def test_quantize_int8_precision(
    input_dtype,
    bits,
    contract_dim,
    block_shape,
    shape,
    layout,
    input_case,
    output_layout,
):
    source = torch.linspace(-500, 500, prod(shape), device="cuda").reshape(shape)
    if input_case == "boundaries":
        qmax = (1 << (bits - 1)) - 1
        midpoints = torch.arange(-qmax, qmax, device="cuda") + 0.5
        pattern = torch.cat(
            (
                source.new_tensor((qmax, -qmax, 0.0, -0.0)),
                torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf)),
                midpoints,
                torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf)),
            )
        )
        source.copy_(
            pattern[torch.arange(prod(shape), device="cuda") % pattern.numel()].reshape(
                shape
            )
        )
    elif input_case == "nonfinite":
        source[..., 0, :3] = source.new_tensor((torch.nan, torch.inf, -torch.inf))
    elif input_case == "tiny":
        source *= 1e-35
        source[..., 0, :] = 0
    source = source.to(input_dtype)
    if layout == "strided":
        source = source.repeat_interleave(2, dim=-1)[..., ::2]
    elif layout == "transposed":
        source = source.mT
    elif layout == "broadcast":
        source = source[..., :1, :].expand(shape)
    elif layout == "offset":
        source = torch.cat((source.new_zeros(1), source.flatten()))[1:].view(shape)
    original = source.clone()

    codes, scales, global_scale = quantize_int8(
        source,
        contract_dim,
        block_shape,
        bits,
        backend="cuda",
        output_layout=output_layout,
    )
    expected_codes, expected_scales, _ = quantize_int8(
        source, contract_dim, block_shape, bits, backend="eager"
    )
    assert torch.equal(codes, expected_codes)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0, equal_nan=True)
    assert global_scale is None
    assert torch.equal(
        source.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)
    )


@cuda_only
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
def test_quantize_int8_stochastic_rounding_precision(
    input_dtype, bits, contract_dim, block_shape
):
    source = torch.full((256, 256), 1.25, device="cuda", dtype=input_dtype)
    source[:, 1::2] = -1.25
    qmax = (1 << (bits - 1)) - 1
    if block_shape == (0, 0):
        source[0, 0] = qmax
    elif block_shape == (1, 0):
        source[:, 0] = qmax
    else:
        source[:: block_shape[0], :: block_shape[1]] = qmax
    if contract_dim == -2:
        source = source.mT
    original = source.clone()

    torch.manual_seed(0)
    initial_state = torch.cuda.get_rng_state()
    rne_codes, rne_scales, _ = quantize_int8(
        source, contract_dim, block_shape, bits, backend="cuda"
    )
    assert torch.equal(torch.cuda.get_rng_state(), initial_state)
    torch.manual_seed(1)
    initial_state = torch.cuda.get_rng_state()
    codes, scales, global_scale = quantize_int8(
        source, contract_dim, block_shape, bits, True, backend="cuda"
    )
    assert not torch.equal(torch.cuda.get_rng_state(), initial_state)
    next_codes, next_scales, _ = quantize_int8(
        source, contract_dim, block_shape, bits, True, backend="cuda"
    )
    torch.manual_seed(1)
    repeated_codes, repeated_scales, _ = quantize_int8(
        source, contract_dim, block_shape, bits, True, backend="cuda"
    )
    assert torch.equal(codes, repeated_codes)
    assert not torch.equal(codes, next_codes)
    assert not torch.equal(codes, rne_codes)
    assert torch.equal(scales, repeated_scales)
    assert torch.equal(scales, next_scales)
    assert torch.equal(scales, rne_scales)
    assert global_scale is None
    assert torch.equal(source, original)
    assert torch.all(codes[source == qmax] == qmax)
    for expected in (-1.25, 1.25):
        rounded = codes[source == expected].float()
        lower = -2 if expected < 0 else 1
        assert torch.all((rounded == lower) | (rounded == lower + 1))
        probability = expected - lower
        # Bound the Bernoulli mean by 4.7 standard errors.
        atol = 4.7 * sqrt(probability * (1 - probability) / rounded.numel())
        torch.testing.assert_close(
            rounded.mean(), rounded.new_tensor(expected), rtol=0, atol=atol
        )


@cuda_only
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
def test_quantize_int8_compile_replay(
    bits, contract_dim, block_shape, stochastic_rounding
):
    source = torch.linspace(-127, 127, 35 * 65, device="cuda", dtype=torch.bfloat16)
    source = source.reshape(35, 65).mT
    original = source.clone()
    torch.compiler.reset()

    def quantize(values):
        return quantize_int8(
            values,
            contract_dim,
            block_shape,
            bits,
            stochastic_rounding,
            backend="cuda",
        )

    compiled = torch.compile(quantize, fullgraph=True)
    compiled(source)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        codes, scales, global_scale = compiled(source)
    graph.replay()
    first_codes, first_scales = codes.clone(), scales.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(source, original)
    assert global_scale is None
    assert torch.equal(scales, first_scales)
    if stochastic_rounding:
        assert not torch.equal(codes, first_codes)
    else:
        expected_codes, expected_scales, _ = quantize_int8(
            source, contract_dim, block_shape, bits, backend="eager"
        )
        assert torch.equal(codes, expected_codes)
        assert torch.equal(scales, expected_scales)


@cuda_only
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("shape", STAT_SHAPES)
@pytest.mark.parametrize("layout", STAT_LAYOUTS)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
def test_quantize_int8_statistics_precision(
    input_dtype,
    bits,
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
    ordinary = quantize_int8(
        source,
        contract_dim,
        block_shape,
        bits,
        stochastic_rounding,
        backend="cuda",
        output_layout=output_layout,
    )
    ordinary_rng = torch.cuda.get_rng_state()
    torch.manual_seed(123)
    codes, scales, global_scale, stats = quantize_int8(
        source,
        contract_dim,
        block_shape,
        bits,
        stochastic_rounding,
        backend="cuda",
        output_layout=output_layout,
        return_quantization_stats=True,
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
    torch.testing.assert_close(
        stats[:2] / energy_scale, expected[:2] / energy_scale, atol=2.3e-6, rtol=0
    )


@cuda_only
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("stochastic_rounding", (False, True))
def test_quantize_int8_statistics_compile(
    contract_dim, output_layout, stochastic_rounding
):
    source = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    bits = 8
    block_shape = (1, 128)

    def quantize(source):
        return quantize_int8(
            source,
            contract_dim,
            block_shape,
            bits,
            stochastic_rounding,
            backend="cuda",
            output_layout=output_layout,
            return_quantization_stats=True,
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
