"""Eager dense quantization-statistics contracts."""

import pytest
import torch

from src.kernel.ops import (
    dequantize_dense,
    quantize_fp8,
    quantize_int8,
    quantize_mxfp8,
    quantize_nvfp4,
)
from src.metrics.quant import accumulate_quantization_sums
from tests.fast.helper import cuda_only


OUTPUT_LAYOUTS = ("row_major", "column_major")
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
CONTRACT_DIMS = (-2, -1)
STOCHASTIC_ROUNDING = (False, True)
BLOCK_SHAPES = ((0, 0), (1, 0), (1, 16), (16, 16))
FORMATS = (
    "fp8_e4m3",
    "fp8_e5m2",
    "mxfp8",
    "int4",
    "int8",
    "fp4_e2m1",
    "fp4_e2m1_4over6",
)
COMPILE_FORMATS = ("fp8_e4m3", "mxfp8", "fp4_e2m1")


def _source(fmt: str, contract_dim: int, dtype: torch.dtype) -> torch.Tensor:
    shape = (19, 48) if fmt.startswith("fp4") else (19, 35)
    if contract_dim == -2:
        shape = shape[::-1]
    generator = torch.Generator(device="cuda").manual_seed(17)
    values = torch.randn(shape, device="cuda", generator=generator) * 3.0
    values[:, ::7] = 0.0
    values[1, 1] = 1e-4
    return values.to(dtype)


def _quantize(
    source: torch.Tensor,
    fmt: str,
    contract_dim: int,
    block_shape: tuple[int, int],
    stochastic_rounding: bool,
    return_quantization_stats: bool,
    output_layout: str = "row_major",
):
    if fmt == "fp8_e4m3":
        return quantize_fp8(
            source,
            contract_dim,
            torch.float8_e4m3fn,
            block_shape,
            stochastic_rounding,
            backend="eager",
            scale_dtype=torch.float8_e4m3fn,
            enable_global_scale=True,
            return_quantization_stats=return_quantization_stats,
            output_layout=output_layout,
        )
    if fmt == "fp8_e5m2":
        return quantize_fp8(
            source,
            contract_dim,
            torch.float8_e5m2,
            block_shape,
            stochastic_rounding,
            backend="eager",
            return_quantization_stats=return_quantization_stats,
            output_layout=output_layout,
        )
    if fmt == "mxfp8":
        return quantize_mxfp8(
            source,
            contract_dim,
            block_shape,
            stochastic_rounding=stochastic_rounding,
            backend="eager",
            return_quantization_stats=return_quantization_stats,
            output_layout=output_layout,
        )
    if fmt.startswith("int"):
        return quantize_int8(
            source,
            contract_dim,
            block_shape,
            bits=int(fmt[3:]),
            stochastic_rounding=stochastic_rounding,
            backend="eager",
            return_quantization_stats=return_quantization_stats,
            output_layout=output_layout,
        )
    return quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        enable_global_scale=True,
        stochastic_rounding=stochastic_rounding,
        backend="eager",
        qmax=4.0 if fmt.endswith("4over6") else 6.0,
        return_quantization_stats=return_quantization_stats,
        output_layout=output_layout,
    )


def _oracle_stats(
    source: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None,
    contract_dim: int,
    block_shape: tuple[int, int],
) -> torch.Tensor:
    dequantized = dequantize_dense(
        codes,
        scale,
        contract_dim,
        block_shape,
        global_scale,
        backend="eager",
    )
    sums = accumulate_quantization_sums(
        source, codes, dequantized, contract_dim=contract_dim
    )
    return torch.stack(tuple(value.reshape(()) for value in sums))


@cuda_only
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", BLOCK_SHAPES)
@pytest.mark.parametrize("stochastic_rounding", STOCHASTIC_ROUNDING)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
def test_quantize_dense_quantization_stats_precision(
    fmt, dtype, contract_dim, block_shape, stochastic_rounding, output_layout
):
    if fmt == "mxfp8" and block_shape not in ((1, 16), (16, 16)):
        pytest.skip("MXFP8 requires a positive 16-or-32-multiple contraction block")
    source = _source(fmt, contract_dim, dtype).requires_grad_()
    original = source.detach().clone()
    torch.manual_seed(29)
    ordinary_codes, ordinary_scale, ordinary_global = _quantize(
        source,
        fmt,
        contract_dim,
        block_shape,
        stochastic_rounding,
        False,
        output_layout,
    )
    torch.manual_seed(29)
    codes, scale, global_scale, stats = _quantize(
        source, fmt, contract_dim, block_shape, stochastic_rounding, True, output_layout
    )

    code_values = codes.view(torch.uint8) if codes.dtype.is_floating_point else codes
    ordinary_code_values = (
        ordinary_codes.view(torch.uint8)
        if ordinary_codes.dtype.is_floating_point
        else ordinary_codes
    )
    scale_values = scale.view(torch.uint8) if scale.dtype.itemsize == 1 else scale
    ordinary_scale_values = (
        ordinary_scale.view(torch.uint8)
        if ordinary_scale.dtype.itemsize == 1
        else ordinary_scale
    )
    assert torch.equal(code_values, ordinary_code_values)
    assert torch.equal(scale_values, ordinary_scale_values)
    if global_scale is None or ordinary_global is None:
        assert global_scale is ordinary_global
    else:
        assert torch.equal(global_scale, ordinary_global)
    assert stats is not None and stats.dtype is torch.float32 and stats.shape == (5,)
    assert not stats.requires_grad
    assert (codes if output_layout == "row_major" else codes.mT).is_contiguous()
    expected = _oracle_stats(
        source, codes, scale, global_scale, contract_dim, block_shape
    )
    normalizer = expected[:2].abs().clamp_min(1.0)
    # The full eager grid peaks at 2.1e-7 from different reduction trees.
    torch.testing.assert_close(
        stats[:2] / normalizer, expected[:2] / normalizer, rtol=0, atol=8.7e-7
    )
    assert torch.equal(stats[2:], expected[2:])
    assert torch.equal(source.detach(), original)


@cuda_only
@pytest.mark.parametrize("fmt", COMPILE_FORMATS)
@pytest.mark.parametrize("stochastic_rounding", STOCHASTIC_ROUNDING)
@pytest.mark.parametrize("output_layout", OUTPUT_LAYOUTS)
def test_quantize_dense_quantization_stats_compile_precision(
    fmt, stochastic_rounding, output_layout
):
    contract_dim, block_shape = -1, (1, 16)
    source = _source(fmt, contract_dim, torch.bfloat16).requires_grad_()

    def quantize(values: torch.Tensor, return_quantization_stats: bool):
        return _quantize(
            values,
            fmt,
            contract_dim,
            block_shape,
            stochastic_rounding,
            return_quantization_stats,
            output_layout,
        )

    torch.compiler.reset()
    try:
        compiled = torch.compile(quantize, fullgraph=True)
        torch.manual_seed(41)
        ordinary_codes, ordinary_scale, ordinary_global = compiled(source, False)
        torch.manual_seed(41)
        codes, scale, global_scale, stats = compiled(source, True)
        code_values = (
            codes.view(torch.uint8) if codes.dtype.is_floating_point else codes
        )
        ordinary_code_values = (
            ordinary_codes.view(torch.uint8)
            if ordinary_codes.dtype.is_floating_point
            else ordinary_codes
        )
        scale_values = scale.view(torch.uint8) if scale.dtype.itemsize == 1 else scale
        ordinary_scale_values = (
            ordinary_scale.view(torch.uint8)
            if ordinary_scale.dtype.itemsize == 1
            else ordinary_scale
        )
        assert torch.equal(code_values, ordinary_code_values)
        assert torch.equal(scale_values, ordinary_scale_values)
        if global_scale is None or ordinary_global is None:
            assert global_scale is ordinary_global
        else:
            assert torch.equal(global_scale, ordinary_global)
        assert stats is not None and not stats.requires_grad
        assert (codes if output_layout == "row_major" else codes.mT).is_contiguous()
        expected = _oracle_stats(
            source, codes, scale, global_scale, contract_dim, block_shape
        )
        normalizer = expected[:2].abs().clamp_min(1.0)
        torch.testing.assert_close(
            stats[:2] / normalizer, expected[:2] / normalizer, rtol=0, atol=8.7e-7
        )
        assert torch.equal(stats[2:], expected[2:])
    finally:
        torch.compiler.reset()
