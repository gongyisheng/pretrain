"""Eager FP8 quantization contracts."""

import pytest
import torch

from src.kernel.ops import quantize_fp8, quantize_fp8_grouped
from tests.fast.helper import cuda_only

CONTRACT_DIMS = (-2, -1)
BLOCK_SHAPES = ((1, 16), (16, 16))


def test_quantize_fp8_stochastic_rounding():
    source = torch.full((64, 32), 1.0625)
    source[:, 0] = 448

    rne_codes, rne_scales, _ = quantize_fp8(
        source, -1, torch.float8_e4m3fn, (1, 32), backend="eager"
    )
    torch.manual_seed(0)
    before = (
        torch.cuda.get_rng_state(source.device)
        if source.is_cuda
        else torch.get_rng_state()
    )
    first_codes, first_scales, _ = quantize_fp8(
        source, -1, torch.float8_e4m3fn, (1, 32), True, backend="eager"
    )
    after = (
        torch.cuda.get_rng_state(source.device)
        if source.is_cuda
        else torch.get_rng_state()
    )
    assert not torch.equal(after, before)
    second_codes, second_scales, _ = quantize_fp8(
        source, -1, torch.float8_e4m3fn, (1, 32), True, backend="eager"
    )

    assert not torch.equal(
        first_codes.view(torch.uint8), second_codes.view(torch.uint8)
    )
    assert torch.equal(first_scales, second_scales)
    assert torch.equal(first_scales, rne_scales)
    assert rne_codes.shape == first_codes.shape


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
def test_quantize_fp8_grouped_precision(contract_dim, block_shape):
    source = torch.linspace(-32, 32, 160).reshape(5, 32)
    if contract_dim == -2:
        source = source.mT
    offs = torch.tensor((16, 16, 32), dtype=torch.int32)
    first_codes, first_scale, _ = quantize_fp8(
        source.narrow(contract_dim, 0, 16),
        contract_dim,
        torch.float8_e4m3fn,
        block_shape,
        backend="eager",
    )
    second_codes, second_scale, _ = quantize_fp8(
        source.narrow(contract_dim, 16, 16),
        contract_dim,
        torch.float8_e4m3fn,
        block_shape,
        backend="eager",
    )

    codes, scale, _ = quantize_fp8_grouped(
        source,
        offs,
        contract_dim,
        contract_dim,
        torch.float8_e4m3fn,
        block_shape,
        backend="eager",
    )
    expected_shape = list(source.shape)
    expected_shape[contract_dim] = 5
    expected_scale = torch.full(expected_shape, 1e-30)
    expected_scale.narrow(contract_dim, 0, 2).copy_(
        torch.cat((first_scale, second_scale), dim=contract_dim)
    )

    assert torch.equal(
        codes.view(torch.uint8),
        torch.cat((first_codes, second_codes), dim=contract_dim).view(torch.uint8),
    )
    assert torch.equal(scale, expected_scale)


@cuda_only
def test_quantize_fp8_grouped_compile():
    source = torch.linspace(-32, 32, 160, device="cuda", dtype=torch.bfloat16).reshape(
        5, 32
    )
    offs = torch.tensor((16, 16, 32), dtype=torch.int32, device="cuda")

    def quantize(values):
        return quantize_fp8_grouped(values, offs, -1, -1, torch.float8_e4m3fn, (1, 16))

    torch.compiler.reset()
    try:
        eager_codes, eager_scale, _ = quantize(source)
        compiled_codes, compiled_scale, compiled_global = torch.compile(
            quantize,
            backend="inductor" if source.is_cuda else "eager",
            fullgraph=True,
        )(source)
    finally:
        torch.compiler.reset()

    assert torch.equal(compiled_codes.view(torch.uint8), eager_codes.view(torch.uint8))
    assert torch.equal(compiled_scale, eager_scale)
    assert compiled_global is None
