"""Eager MXFP8 quantization contracts."""

from itertools import product
from math import ceil, log2, prod

import pytest
import torch

from src.kernel.ops import quantize_mxfp8, quantize_mxfp8_grouped
from src.kernel.utils import to_swizzle_32_4_4
from tests.fast.helper import cuda_only


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
PRECISION_SHAPES = ((17, 35), (2, 17, 35))
COMPILE_BLOCK_SHAPES = ((1, 32), (64, 64))
COMPILE_SHAPES = ((131, 259), (2, 131, 259))
STOCHASTIC_ROUNDING = (False, True)
OUTPUT_LAYOUTS = ("row_major", "column_major")
RAGGED_DIMS = (-2, -1)
SCALE_LAYOUT_SHAPES = ((128, 128), (130, 160))


def _transposed_source(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    source = (
        torch.linspace(-500, 500, prod(shape), dtype=torch.float32, device=device)
        .reshape(*shape[:-2], shape[-1], shape[-2])
        .transpose(-2, -1)
    )
    source[..., 0, 0] = -0.0
    source[..., 0, 1] = 448.0
    return source.to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("shape", PRECISION_SHAPES)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
def test_quantize_mxfp8_precision(
    dtype, contract_dim, block_shape, shape, output_layout
):
    """Check transposed tails against an independent block-loop oracle."""
    source = _transposed_source(shape, dtype, torch.device("cpu"))
    values = source.float().movedim(contract_dim, -1)
    outer, contraction = values.shape[-2:]
    block_outer, block_size = block_shape
    expected_codes = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    expected_scale_bytes = torch.empty(
        *values.shape[:-2],
        outer,
        ceil(contraction / block_size),
        dtype=torch.uint8,
        device="cpu",
    )
    qmax = torch.finfo(torch.float8_e4m3fn).max
    for prefix in product(*(range(size) for size in values.shape[:-2])):
        for outer_start in range(0, outer, block_outer):
            outer_stop = min(outer_start + block_outer, outer)
            for contract_start in range(0, contraction, block_size):
                contract_stop = min(contract_start + block_size, contraction)
                tile = values[
                    prefix
                    + (
                        slice(outer_start, outer_stop),
                        slice(contract_start, contract_stop),
                    )
                ]
                maximum = float(tile.abs().amax())
                scale_code = (
                    0
                    if maximum == 0
                    else max(0, min(254, ceil(log2(maximum / qmax)) + 127))
                )
                scale = 2.0 ** (scale_code - 127)
                expected_codes[
                    prefix
                    + (
                        slice(outer_start, outer_stop),
                        slice(contract_start, contract_stop),
                    )
                ] = (tile / scale).clamp(-qmax, qmax).to(torch.float8_e4m3fn)
                expected_scale_bytes[
                    prefix
                    + (
                        slice(outer_start, outer_stop),
                        slice(
                            contract_start // block_size,
                            contract_start // block_size + 1,
                        ),
                    )
                ] = scale_code

    actual_codes, actual_scales, _ = quantize_mxfp8(
        source, contract_dim, block_shape, backend="eager", output_layout=output_layout
    )
    expected_codes = expected_codes.movedim(-1, contract_dim).contiguous()
    expected_scales = (
        expected_scale_bytes.movedim(-1, contract_dim)
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )

    assert torch.equal(actual_codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(
        actual_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )


@pytest.mark.parametrize("shape", SCALE_LAYOUT_SHAPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_outer", (1, 32))
def test_quantize_mxfp8_scale_layout(shape, contract_dim, block_outer):
    source = torch.linspace(-448, 448, prod(shape), dtype=torch.float32).reshape(shape)
    codes, scales, _ = quantize_mxfp8(
        source,
        contract_dim,
        (block_outer, 32),
        backend="eager",
    )
    packed_codes, packed_scales, _ = quantize_mxfp8(
        source,
        contract_dim,
        (block_outer, 32),
        backend="eager",
        scale_layout="swizzled_32_4_4",
    )

    expected_scales = to_swizzle_32_4_4(scales if contract_dim == -1 else scales.t())
    assert torch.equal(packed_codes.view(torch.uint8), codes.view(torch.uint8))
    assert packed_scales.ndim == 1
    assert torch.equal(
        packed_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )


def test_quantize_mxfp8_grouped_resets_outer_tiles():
    source = torch.zeros(9, 32)
    source[:4, 0] = 448
    source[4:, 0] = 224
    offs = torch.tensor((4, 4, 9), dtype=torch.int32)

    _, scales, _ = quantize_mxfp8_grouped(
        source, offs, -2, -1, (16, 16), backend="eager"
    )

    scale_bytes = scales.view(torch.uint8)
    assert torch.equal(scale_bytes[:4, 0], torch.full((4,), 127, dtype=torch.uint8))
    assert torch.equal(scale_bytes[4:, 0], torch.full((5,), 126, dtype=torch.uint8))


@cuda_only
@pytest.mark.parametrize("shape", COMPILE_SHAPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize("stochastic_rounding", STOCHASTIC_ROUNDING)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
def test_quantize_mxfp8_compile(
    shape, contract_dim, block_shape, stochastic_rounding, output_layout
):
    source = _transposed_source(shape, torch.bfloat16, torch.device("cuda"))

    def quantize(
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return quantize_mxfp8(
            values,
            contract_dim,
            block_shape,
            stochastic_rounding=stochastic_rounding,
            backend="eager",
            output_layout=output_layout,
        )

    torch.compiler.reset()
    try:
        compiled = torch.compile(quantize, fullgraph=True)
        if not stochastic_rounding:
            eager_codes, eager_scales, _ = quantize(source)
            compiled_codes, compiled_scales, _ = compiled(source)
            assert torch.equal(
                compiled_codes.view(torch.uint8), eager_codes.view(torch.uint8)
            )
            assert torch.equal(
                compiled_scales.view(torch.uint8), eager_scales.view(torch.uint8)
            )
            return

        source.fill_(1.0625)
        source.select(contract_dim, 0).fill_(448.0)
        rne_codes, rne_scales, _ = quantize_mxfp8(
            source, contract_dim, block_shape, backend="eager"
        )
        torch.manual_seed(0)
        before = torch.cuda.get_rng_state()
        first_codes, first_scales, _ = compiled(source)
        assert not torch.equal(torch.cuda.get_rng_state(), before)
        second_codes, second_scales, _ = compiled(source)
        torch.manual_seed(0)
        repeated_codes, repeated_scales, _ = compiled(source)

        assert not torch.equal(
            first_codes.view(torch.uint8), second_codes.view(torch.uint8)
        )
        assert torch.equal(
            first_codes.view(torch.uint8), repeated_codes.view(torch.uint8)
        )
        assert torch.equal(
            first_scales.view(torch.uint8), repeated_scales.view(torch.uint8)
        )
        assert torch.equal(
            first_scales.view(torch.uint8), second_scales.view(torch.uint8)
        )
        assert torch.equal(first_scales.view(torch.uint8), rne_scales.view(torch.uint8))
        assert rne_codes.shape == first_codes.shape
    finally:
        torch.compiler.reset()


@cuda_only
@pytest.mark.parametrize("stochastic_rounding", STOCHASTIC_ROUNDING)
@pytest.mark.parametrize("ragged_dim", RAGGED_DIMS)
def test_quantize_mxfp8_grouped_compile(stochastic_rounding, ragged_dim):
    source = torch.full((64, 64), 1.0625, device="cuda", dtype=torch.bfloat16)
    source[:, ::16] = 448.0
    offs = torch.tensor((16, 16, 64), dtype=torch.int32, device="cuda")

    def quantize(values):
        return quantize_mxfp8_grouped(
            values, offs, ragged_dim, -1, (16, 16), stochastic_rounding
        )

    torch.compiler.reset()
    try:
        eager_codes, eager_scales, _ = quantize(source)
        compiled = torch.compile(quantize, fullgraph=True)
        codes, scales, _ = compiled(source)
        assert torch.equal(scales.view(torch.uint8), eager_scales.view(torch.uint8))
        if stochastic_rounding:
            again, again_scales, _ = compiled(source)
            assert not torch.equal(codes.view(torch.uint8), again.view(torch.uint8))
            assert torch.equal(scales.view(torch.uint8), again_scales.view(torch.uint8))
            rounded = codes[source != 448.0].float()
            assert torch.all((rounded == 1.0) | (rounded == 1.125))
        else:
            assert torch.equal(codes.view(torch.uint8), eager_codes.view(torch.uint8))
    finally:
        torch.compiler.reset()
