"""Eager NVFP4 quantization contracts."""

import pytest
import torch

from src.kernel.ops import quantize_nvfp4, quantize_nvfp4_grouped
from src.quant.constants import _FP4_E2M1_VALUES
from src.kernel.utils import to_swizzle_32_4_4
from tests.fast.helper import cuda_only

CONTRACT_DIMS = (-2, -1)
BLOCK_SHAPES = ((1, 16), (16, 16))
STOCHASTIC_ROUNDING = (False, True)
RAGGED_DIMS = (-2, -1)
RECIPE_BLOCK_SHAPES = ((0, 0), (1, 0), (1, 8))
RECIPE_SCALE_DTYPES = (torch.float32, torch.float8_e4m3fn)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
MIDPOINTS = tuple(
    (lower + upper) / 2 for lower, upper in zip(_FP4_E2M1_VALUES, _FP4_E2M1_VALUES[1:])
)
PRECISION_CASES = (
    ("values", (0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE)),
    ("midpoints", (0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE)),
    ("below", (0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE)),
    ("above", (0x21, 0x43, 0x65, 0x77, 0xA9, 0xCB, 0xED, 0xFF)),
)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("case", PRECISION_CASES)
@pytest.mark.parametrize("grouped", (False, True))
def test_quantize_nvfp4_precision(dtype, contract_dim, case, grouped):
    case, packed_bytes = case
    if case == "values":
        values = torch.tensor(_FP4_E2M1_VALUES, dtype=dtype)
    else:
        values = torch.tensor(MIDPOINTS, dtype=dtype)
        if case != "midpoints":
            direction = -torch.inf if case == "below" else torch.inf
            values = torch.nextafter(values, torch.full_like(values, direction))
        values = torch.cat((values, values.new_tensor((6.0,))))
    source = torch.cat((values, -values)).repeat(2, 1)
    expected = torch.tensor(packed_bytes, dtype=torch.uint8).repeat(2, 1)
    if contract_dim == -2:
        source, expected = source.mT, expected.mT
    if grouped:
        codes, scales, global_scale = quantize_nvfp4_grouped(
            source,
            torch.tensor((1, 2), dtype=torch.int32),
            -1 if contract_dim == -2 else -2,
            contract_dim,
            (1, 0),
            False,
            backend="eager",
            scale_dtype=torch.float32,
        )
    else:
        codes, scales, global_scale, _ = quantize_nvfp4(
            source,
            contract_dim,
            (1, 0),
            scale_dtype=torch.float32,
            enable_global_scale=False,
            backend="eager",
        )

    assert torch.equal(codes, expected)
    assert torch.equal(scales, torch.ones_like(scales))
    assert global_scale is None


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
def test_quantize_nvfp4_grouped_precision(contract_dim, block_shape):
    source = torch.linspace(-32, 32, 160).reshape(5, 32)
    if contract_dim == -2:
        source = source.mT
    offs = torch.tensor((16, 16, 32), dtype=torch.int32)
    first_codes, first_scale, first_global, _ = quantize_nvfp4(
        source.narrow(contract_dim, 0, 16), contract_dim, block_shape, backend="eager"
    )
    second_codes, second_scale, second_global, _ = quantize_nvfp4(
        source.narrow(contract_dim, 16, 16), contract_dim, block_shape, backend="eager"
    )
    packed, scales, global_scale = quantize_nvfp4_grouped(
        source,
        offs,
        contract_dim,
        contract_dim,
        block_shape,
        True,
        backend="eager",
    )
    expected_shape = list(source.shape)
    expected_shape[contract_dim] = 5
    expected_scale = torch.full(expected_shape, 2.0**-9).to(torch.float8_e4m3fn)
    expected_scale.narrow(contract_dim, 0, 2).copy_(
        torch.cat((first_scale, second_scale), dim=contract_dim)
    )
    expected_global = torch.cat((first_global, torch.full((1,), 1e-30), second_global))

    assert torch.equal(packed, torch.cat((first_codes, second_codes), dim=contract_dim))
    assert torch.equal(scales.view(torch.uint8), expected_scale.view(torch.uint8))
    assert torch.equal(global_scale, expected_global)


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
def test_quantize_nvfp4_scale_layout(contract_dim, block_shape):
    shape = (129, 64) if contract_dim == -1 else (64, 129)
    source = torch.linspace(-6, 6, 129 * 64).reshape(shape)
    codes, scales, _, stats = quantize_nvfp4(
        source, contract_dim, block_shape, backend="eager"
    )
    packed_codes, packed_scales, _, packed_stats = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        scale_layout="swizzled_32_4_4",
        backend="eager",
    )

    assert stats is False
    assert packed_stats is False
    assert torch.equal(packed_codes, codes)
    assert torch.equal(
        packed_scales.view(torch.uint8),
        to_swizzle_32_4_4(scales if contract_dim == -1 else scales.t()).view(
            torch.uint8
        ),
    )


def test_quantize_nvfp4_grouped_stochastic_rounding():
    source = torch.full((5, 32), 1.25)
    source[:, ::16] = 6
    offs = torch.tensor((16, 16, 32), dtype=torch.int32)

    rne = quantize_nvfp4_grouped(source, offs, -1, -1, (1, 16), True, backend="eager")
    torch.manual_seed(0)
    before = (
        torch.cuda.get_rng_state(source.device)
        if source.is_cuda
        else torch.get_rng_state()
    )
    first = quantize_nvfp4_grouped(
        source, offs, -1, -1, (1, 16), True, True, backend="eager"
    )
    after = (
        torch.cuda.get_rng_state(source.device)
        if source.is_cuda
        else torch.get_rng_state()
    )
    assert not torch.equal(after, before)
    second = quantize_nvfp4_grouped(
        source, offs, -1, -1, (1, 16), True, True, backend="eager"
    )

    assert not torch.equal(first[0], second[0])
    assert torch.equal(first[1].view(torch.uint8), second[1].view(torch.uint8))
    assert torch.equal(first[1].view(torch.uint8), rne[1].view(torch.uint8))


@pytest.mark.parametrize("block_shape", RECIPE_BLOCK_SHAPES)
@pytest.mark.parametrize("scale_dtype", RECIPE_SCALE_DTYPES)
@pytest.mark.parametrize("grouped", (False, True))
def test_quantize_nvfp4_qmax4_precision(block_shape, scale_dtype, grouped):
    source = torch.tensor(((-1792.0, -896.0, 0.0, 1792.0) * 4,) * 8)
    source = source.repeat_interleave(2, dim=-1)[..., ::2]
    expected_codes = torch.tensor(((0xCE, 0x60) * 4,) * 8, dtype=torch.uint8)
    if grouped:
        codes, scales, global_scale = quantize_nvfp4_grouped(
            source,
            torch.tensor((8, 8, 16), dtype=torch.int32),
            -1,
            -1,
            block_shape,
            backend="eager",
            enable_global_scale=True,
            scale_dtype=scale_dtype,
            qmax=4.0,
        )
    else:
        codes, scales, global_scale, _ = quantize_nvfp4(
            source,
            -1,
            block_shape,
            fmt="fp4_e2m1_4over6",
            scale_dtype=scale_dtype,
            backend="eager",
        )

    assert not source.is_contiguous()
    assert torch.equal(codes, expected_codes)
    assert scales.dtype is scale_dtype
    if scale_dtype is torch.float8_e4m3fn:
        expected_global = torch.tensor((1.0, 1e-30, 1.0)) if grouped else torch.ones(1)
        assert torch.equal(global_scale, expected_global)
    else:
        assert global_scale is None


@cuda_only
@pytest.mark.parametrize("stochastic_rounding", STOCHASTIC_ROUNDING)
@pytest.mark.parametrize("ragged_dim", RAGGED_DIMS)
def test_quantize_nvfp4_grouped_compile(stochastic_rounding, ragged_dim):
    source = torch.full((32, 32), 1.3, device="cuda", dtype=torch.bfloat16)
    source[:, ::16] = 6.0
    offs = torch.tensor((16, 16, 32), dtype=torch.int32, device="cuda")

    def quantize(values):
        return quantize_nvfp4_grouped(
            values, offs, ragged_dim, -1, (16, 16), True, stochastic_rounding
        )

    torch.compiler.reset()
    try:
        eager_codes, eager_scales, eager_global = quantize(source)
        compiled = torch.compile(quantize, fullgraph=True)
        codes, scales, global_scale = compiled(source)
        assert torch.equal(scales.view(torch.uint8), eager_scales.view(torch.uint8))
        assert torch.equal(global_scale, eager_global)
        if stochastic_rounding:
            again, again_scales, again_global = compiled(source)
            assert not torch.equal(codes, again)
            assert torch.equal(scales.view(torch.uint8), again_scales.view(torch.uint8))
            assert torch.equal(global_scale, again_global)
        else:
            assert torch.equal(codes, eager_codes)
    finally:
        torch.compiler.reset()
