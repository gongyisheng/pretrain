"""Eager integer quantization contracts."""

import pytest
import torch

from src.kernel.ops import quantize_int8, quantize_int8_grouped
from tests.fast.helper import cuda_only


BITS = (4, 5, 6, 7, 8)
BLOCK_SHAPES = ((0, 0), (1, 0), (1, 8), (8, 8))
CONTRACT_DIMS = (-2, -1)


def _reference(x, contract_dim, block_shape, bits):
    values = x.float().movedim(contract_dim, -1)
    outer, contraction = values.shape[-2:]
    block_outer = block_shape[0] or outer
    block_contract = block_shape[1] or contraction
    qmax = (1 << (bits - 1)) - 1
    codes = torch.empty_like(values, dtype=torch.int8)
    scales = torch.empty(
        outer,
        (contraction + block_contract - 1) // block_contract,
        dtype=torch.float32,
    )
    for outer_start in range(0, outer, block_outer):
        outer_stop = min(outer_start + block_outer, outer)
        for contract_start in range(0, contraction, block_contract):
            contract_stop = min(contract_start + block_contract, contraction)
            tile = values[outer_start:outer_stop, contract_start:contract_stop]
            scale = tile.abs().amax().div(qmax).clamp_min(1e-30)
            codes[outer_start:outer_stop, contract_start:contract_stop] = (
                torch.round(tile / scale).clamp(-qmax, qmax).to(torch.int8)
            )
            scales[outer_start:outer_stop, contract_start // block_contract] = scale
    return codes.movedim(-1, contract_dim).contiguous(), scales.movedim(
        -1, contract_dim
    )


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
def test_quantize_int8_precision(bits, block_shape, contract_dim):
    source = torch.linspace(-23, 19, 91).reshape(7, 13).mT
    expected_codes, expected_scales = _reference(
        source, contract_dim, block_shape, bits
    )

    codes, scales, _ = quantize_int8(
        source, contract_dim, block_shape, bits, backend="eager"
    )

    assert torch.equal(codes, expected_codes)
    assert torch.equal(scales, expected_scales)


def test_quantize_int8_stochastic_rounding():
    source = torch.full((64, 16), 1.5)
    source[:, 0] = 7

    rne_codes, rne_scales, _ = quantize_int8(source, -1, (1, 16), 4, backend="eager")
    torch.manual_seed(0)
    before = (
        torch.cuda.get_rng_state(source.device)
        if source.is_cuda
        else torch.get_rng_state()
    )
    first_codes, first_scales, _ = quantize_int8(
        source, -1, (1, 16), 4, True, backend="eager"
    )
    after = (
        torch.cuda.get_rng_state(source.device)
        if source.is_cuda
        else torch.get_rng_state()
    )
    assert not torch.equal(after, before)
    second_codes, second_scales, _ = quantize_int8(
        source, -1, (1, 16), 4, True, backend="eager"
    )

    assert not torch.equal(first_codes, second_codes)
    assert torch.equal(first_scales, second_scales)
    assert torch.equal(first_scales, rne_scales)
    assert rne_codes.shape == first_codes.shape


def test_quantize_int8_grouped_precision():
    source = torch.linspace(-32, 32, 160).reshape(5, 32)
    offs = torch.tensor((16, 16, 32), dtype=torch.int32)
    first_codes, first_scale, _ = quantize_int8(
        source[:, :16], -1, (1, 16), 6, backend="eager"
    )
    second_codes, second_scale, _ = quantize_int8(
        source[:, 16:], -1, (1, 16), 6, backend="eager"
    )

    codes, scale, _ = quantize_int8_grouped(
        source, offs, -1, -1, (1, 16), 6, backend="eager"
    )
    expected_scale = torch.full((5, 5), 1e-30)
    expected_scale[:, :2] = torch.cat((first_scale, second_scale), dim=-1)

    assert torch.equal(codes, torch.cat((first_codes, second_codes), dim=-1))
    assert torch.equal(scale, expected_scale)


@cuda_only
def test_quantize_int8_grouped_compile():
    source = torch.linspace(-32, 32, 160, device="cuda", dtype=torch.bfloat16).reshape(
        5, 32
    )
    offs = torch.tensor((16, 16, 32), dtype=torch.int32, device="cuda")

    def quantize(values):
        return quantize_int8_grouped(values, offs, -1, -1, (1, 16), 6)

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

    assert torch.equal(compiled_codes, eager_codes)
    assert torch.equal(compiled_scale, eager_scale)
    assert compiled_global is None
