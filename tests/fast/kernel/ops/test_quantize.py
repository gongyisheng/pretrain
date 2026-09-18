"""Public quantization operation contracts."""

from math import prod

import pytest
import torch

from src.kernel.ops import (
    dequantize_dense,
    dequantize_grouped,
    quantize_fp8,
    quantize_fp8_grouped,
    quantize_int8,
    quantize_int8_grouped,
    quantize_mxfp8,
    quantize_mxfp8_grouped,
    quantize_nvfp4,
    quantize_nvfp4_grouped,
    unpack_e2m1,
)
from src.quant.utils import str_to_dtype


FP8_INPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
FP8_CONTRACT_DIMS = (-2, -1)
FP8_BLOCK_SHAPES = ((0, 0), (1, 0), (1, 7), (1, 32), (32, 32))
FP8_SHAPES = ((35, 65), (2, 35, 65))
FP8_RECIPE_SCALE_DTYPES = (torch.float32, torch.float8_e4m3fn)
DENSE_OUTPUT_LAYOUTS = ("row_major", "column_major")
FP8_ERROR_CASES = (
    ((0, 32), torch.float32, -1, torch.float8_e4m3fn, (0, 0), False),
    ((32, 0), torch.float32, -1, torch.float8_e4m3fn, (0, 0), False),
    ((32,), torch.float32, -1, torch.float8_e4m3fn, (0, 0), False),
    ((1, 2, 32, 32), torch.float32, -1, torch.float8_e4m3fn, (0, 0), False),
    ((32, 32), torch.int32, -1, torch.float8_e4m3fn, (0, 0), False),
    ((32, 32), torch.float32, 0, torch.float8_e4m3fn, (0, 0), False),
    ((32, 32), torch.float32, -1, torch.float8_e8m0fnu, (0, 0), False),
    ((32, 32), torch.float32, -1, torch.float8_e4m3fn, (0, 1), False),
    ((32, 32), torch.float32, -1, torch.float8_e4m3fn, (1, -1), False),
    ((32, 32), torch.float32, -1, torch.float8_e4m3fn, (2, 32), False),
    ((32, 32), torch.float32, -1, torch.float8_e4m3fn, (32, 16), False),
    ((32, 32), torch.float32, -1, torch.float8_e4m3fn, (0,), False),
    ((32, 32), torch.float32, -1, torch.float8_e4m3fn, (0, 0, 0), False),
    ((32, 32), torch.float32, -1, torch.float8_e4m3fn, ("0", "0"), False),
    ((32, 32), torch.float32, -1, torch.float8_e4m3fn, (False, 0), False),
    ((32, 32), torch.float32, -1, torch.float8_e4m3fn, (0, 0), 1),
)


@pytest.mark.parametrize("input_dtype", FP8_INPUT_DTYPES)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES)
@pytest.mark.parametrize("contract_dim", FP8_CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", FP8_BLOCK_SHAPES)
@pytest.mark.parametrize("shape", FP8_SHAPES)
def test_quantize_fp8(input_dtype, fp8_dtype, contract_dim, block_shape, shape):
    source = (
        torch.arange(prod(shape), dtype=torch.float32).reshape(shape) / 13 - 40
    ).to(input_dtype)
    original = source.clone()

    codes, scales, global_scale = quantize_fp8(
        source, contract_dim, fp8_dtype, block_shape, backend="eager"
    )

    expected_scale_shape = list(shape)
    block_contract = block_shape[1]
    expected_scale_shape[contract_dim] = (
        (shape[contract_dim] + block_contract - 1) // block_contract
        if block_contract
        else 1
    )
    assert codes.shape == source.shape
    assert codes.dtype is fp8_dtype
    assert codes.is_contiguous()
    assert scales.shape == tuple(expected_scale_shape)
    assert scales.dtype is torch.float32
    assert global_scale is None
    if block_shape == (0, 0):
        assert 0 in scales.stride()
    else:
        assert scales.is_contiguous()
    assert torch.equal(source, original)


@pytest.mark.parametrize("block_shape", ((0, 0), (1, 0), (1, 8)))
@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize("scale_dtype", FP8_RECIPE_SCALE_DTYPES)
@pytest.mark.parametrize("enable_global_scale", (False, True))
def test_quantize_fp8_recipe(block_shape, grouped, scale_dtype, enable_global_scale):
    source = torch.zeros(8, 16)
    if grouped:
        result = quantize_fp8_grouped(
            source,
            torch.tensor((8, 8, 16), dtype=torch.int32),
            -1,
            -1,
            torch.float8_e4m3fn,
            block_shape,
            backend="eager",
            scale_dtype=scale_dtype,
            enable_global_scale=enable_global_scale,
        )
    else:
        result = quantize_fp8(
            source,
            -1,
            torch.float8_e4m3fn,
            block_shape,
            backend="eager",
            scale_dtype=scale_dtype,
            enable_global_scale=enable_global_scale,
        )
    codes, scales, global_scale = result

    assert codes.shape == source.shape
    assert codes.dtype is torch.float8_e4m3fn
    assert scales.dtype is scale_dtype
    if scale_dtype is torch.float8_e4m3fn and enable_global_scale:
        assert global_scale is not None
        assert global_scale.shape == ((3,) if grouped else (1,))
        return
    assert global_scale is None
    if scale_dtype is torch.float8_e4m3fn:
        return
    if grouped:
        without_global = quantize_fp8_grouped(
            source,
            torch.tensor((8, 8, 16), dtype=torch.int32),
            -1,
            -1,
            torch.float8_e4m3fn,
            block_shape,
            backend="eager",
            scale_dtype=scale_dtype,
            enable_global_scale=False,
        )
    else:
        without_global = quantize_fp8(
            source,
            -1,
            torch.float8_e4m3fn,
            block_shape,
            backend="eager",
            scale_dtype=scale_dtype,
            enable_global_scale=False,
        )
    assert without_global[2] is None
    assert torch.equal(codes.view(torch.uint8), without_global[0].view(torch.uint8))
    assert torch.equal(scales, without_global[1])


@pytest.mark.parametrize("case", FP8_ERROR_CASES)
def test_quantize_fp8_raise_error(case):
    shape, input_dtype, contract_dim, fp8_dtype, block_shape, stochastic_rounding = case
    with pytest.raises(ValueError):
        quantize_fp8(
            torch.empty(shape, dtype=input_dtype),
            contract_dim,
            fp8_dtype,
            block_shape,
            stochastic_rounding,
            backend="eager",
        )


INT8_BITS = (4, 5, 6, 7, 8)
INT8_BLOCK_SHAPES = ((0, 0), (1, 0), (1, 8), (8, 8))
INT8_CONTRACT_DIMS = (-2, -1)
INT8_INPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
INT8_SHAPES = ((13, 7), (2, 13, 7))
INT8_BACKENDS = (None, "eager", "cuda")
INT8_LAYOUTS = ("dense", "transposed", "strided", "broadcast")
INT8_RECIPE_SCALE_DTYPES = (torch.float32, torch.float8_e4m3fn)
INT8_ERROR_CASES = (
    ((2, 8), torch.float32, -1, (1, 8), 3),
    ((2, 8), torch.float32, -1, (1, 8), 9),
    ((2, 8), torch.float32, 0, (1, 8), 8),
    ((2, 8), torch.float32, -1, (2, 8), 8),
    ((2, 8), torch.float32, -1, (1, -8), 8),
    ((2, 8), torch.float32, -1, (1,), 8),
    ((2, 8), torch.float32, -1, (1, 8), True),
    ((2, 8), torch.float32, -1, (1, 8), 8.0),
    ((2, 8), torch.int32, -1, (1, 8), 8),
)


@pytest.mark.parametrize("bits", INT8_BITS)
@pytest.mark.parametrize("block_shape", INT8_BLOCK_SHAPES)
@pytest.mark.parametrize("contract_dim", INT8_CONTRACT_DIMS)
@pytest.mark.parametrize("input_dtype", INT8_INPUT_DTYPES)
@pytest.mark.parametrize("shape", INT8_SHAPES)
@pytest.mark.parametrize("backend", INT8_BACKENDS)
@pytest.mark.parametrize("layout", INT8_LAYOUTS)
def test_quantize_int8(
    bits, block_shape, contract_dim, input_dtype, shape, backend, layout
):
    source = torch.zeros(shape, dtype=input_dtype)
    if backend == "cuda" and not source.is_cuda:
        pytest.skip("CUDA requires a CUDA device")
    if layout == "transposed":
        source = source.mT
    elif layout == "strided":
        source = source.repeat_interleave(2, dim=-1)[..., ::2]
    elif layout == "broadcast":
        source = source[..., :1, :].expand(shape)
    codes, scales, global_scale = quantize_int8(
        source, contract_dim, block_shape, bits, backend=backend
    )

    assert codes.dtype is torch.int8
    assert codes.is_contiguous()
    assert scales.dtype is torch.float32
    assert global_scale is None
    assert codes.shape == source.shape
    expected_scale_shape = list(source.shape)
    expected_scale_shape[contract_dim] = (
        (source.shape[contract_dim] + block_shape[1] - 1) // block_shape[1]
        if block_shape[1]
        else 1
    )
    assert scales.shape == tuple(expected_scale_shape)
    if block_shape == (0, 0):
        assert 0 in scales.stride()
    else:
        assert scales.is_contiguous()


@pytest.mark.parametrize("block_shape", ((0, 0), (1, 0), (1, 8)))
@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize("scale_dtype", INT8_RECIPE_SCALE_DTYPES)
@pytest.mark.parametrize("enable_global_scale", (False, True))
@pytest.mark.parametrize("backend", (None, "eager"))
def test_quantize_int8_recipe(
    block_shape, grouped, scale_dtype, enable_global_scale, backend
):
    source = torch.zeros(8, 16)
    if grouped:
        result = quantize_int8_grouped(
            source,
            torch.tensor((8, 8, 16), dtype=torch.int32),
            -1,
            -1,
            block_shape,
            4,
            backend=backend,
            scale_dtype=scale_dtype,
            enable_global_scale=enable_global_scale,
        )
    else:
        result = quantize_int8(
            source,
            -1,
            block_shape,
            4,
            backend=backend,
            scale_dtype=scale_dtype,
            enable_global_scale=enable_global_scale,
        )
    codes, scales, global_scale = result

    assert codes.shape == source.shape
    assert codes.dtype is torch.int8
    assert scales.dtype is scale_dtype
    if scale_dtype is torch.float8_e4m3fn and enable_global_scale:
        assert global_scale is not None
        assert global_scale.shape == ((3,) if grouped else (1,))
        return
    assert global_scale is None
    if scale_dtype is torch.float8_e4m3fn:
        return
    if grouped:
        without_global = quantize_int8_grouped(
            source,
            torch.tensor((8, 8, 16), dtype=torch.int32),
            -1,
            -1,
            block_shape,
            4,
            backend="eager",
            scale_dtype=scale_dtype,
            enable_global_scale=False,
        )
    else:
        without_global = quantize_int8(
            source,
            -1,
            block_shape,
            4,
            backend="eager",
            scale_dtype=scale_dtype,
            enable_global_scale=False,
        )
    assert without_global[2] is None
    assert torch.equal(codes, without_global[0])
    assert torch.equal(scales, without_global[1])


@pytest.mark.parametrize("case", INT8_ERROR_CASES)
def test_quantize_int8_raise_error(case):
    shape, dtype, contract_dim, block_shape, bits = case
    source = torch.empty(shape, dtype=dtype)
    with pytest.raises(ValueError):
        quantize_int8(source, contract_dim, block_shape, bits, backend="eager")


NVFP4_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
NVFP4_CONTRACT_DIMS = (-2, -1)
NVFP4_BLOCK_SHAPES = ((1, 16), (16, 16))
NVFP4_GLOBAL_SCALES = (False, True)
NVFP4_LAYOUTS = ("dense", "strided")
NVFP4_SHAPES = ((32, 64), (2, 32, 64))
NVFP4_ERROR_CASES = (
    ((16,), torch.float32, -1, (1, 16)),
    ((1, 16, 16, 16), torch.float32, -1, (1, 16)),
    ((0, 16), torch.float32, -1, (1, 16)),
    ((16, 0), torch.float32, -1, (1, 16)),
    ((16, 15), torch.float32, -1, (1, 16)),
    ((16, 16), torch.float32, 0, (1, 16)),
    ((16, 16), torch.int32, -1, (1, 16)),
    ((16, 16), torch.float32, -1, (2, 32)),
)
NVFP4_RECIPE_BLOCK_SHAPES = ((0, 0), (1, 0), (1, 8))
NVFP4_RECIPE_SCALE_DTYPES = (torch.float32, torch.float8_e4m3fn)

MXFP8_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
MXFP8_FORMATS = ("fp8_e4m3",)
MXFP8_CONTRACT_DIMS = (-2, -1)
MXFP8_BLOCK_SHAPES = (
    (1, 16),
    (1, 32),
    (1, 64),
    (1, 128),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
)
MXFP8_SHAPES = ((35, 65), (2, 35, 65))
MXFP8_LAYOUTS = ("dense", "strided", "broadcast")
MXFP8_OUTPUT_LAYOUTS = ("row_major", "column_major")
MXFP8_SWIZZLE_EMPTY_CASES = (
    ((0, 32), -2),
    ((0, 32), -1),
    ((32, 0), -2),
    ((32, 0), -1),
)
MXFP8_DEVICES = ("cpu", "cuda")
MXFP8_BACKENDS = ("eager", "cuda")
MXFP8_EMPTY_CASES = (
    ((0, 32), -2),
    ((0, 32), -1),
    ((32, 0), -2),
    ((32, 0), -1),
    ((2, 0, 32), -2),
    ((2, 32, 0), -1),
)
MXFP8_ERROR_CASES = (
    ((), -1, (1, 32), torch.float32, "fp8_e4m3", False, "row_major"),
    ((32,), -1, (1, 32), torch.float32, "fp8_e4m3", False, "row_major"),
    ((2, 32), 0, (1, 32), torch.float32, "fp8_e4m3", False, "row_major"),
    ((2, 32), -3, (1, 32), torch.float32, "fp8_e4m3", False, "row_major"),
    ((2, 32), -1, (1, 0), torch.float32, "fp8_e4m3", False, "row_major"),
    ((2, 32), -1, (1, 24), torch.float32, "fp8_e4m3", False, "row_major"),
    ((2, 32), -1, (2, 32), torch.float32, "fp8_e4m3", False, "row_major"),
    ((2, 32), -1, (32, 64), torch.float32, "fp8_e4m3", False, "row_major"),
    ((2, 32), -1, (1, 32), torch.int32, "fp8_e4m3", False, "row_major"),
    ((2, 32), -1, (1, 32), torch.float32, "fp8_e8m0", False, "row_major"),
    ((2, 32), -1, (1, 32), torch.float32, "fp8_e5m2", False, "row_major"),
    ((2, 32), -1, (1, 32), torch.float32, "fp8_e4m3", 1, "row_major"),
    ((2, 32), -1, (1, 32), torch.float32, "fp8_e4m3", False, "invalid"),
)


def _make_nvfp4_source(shape, dtype, layout):
    source = torch.arange(prod(shape), dtype=torch.float32).reshape(shape) / 19 - 3
    if layout == "strided":
        source = source.mT.contiguous().mT
    return source.to(dtype)


def test_dequantize_dense_raise_error():
    with pytest.raises(ValueError):
        dequantize_dense(
            torch.empty(2, 4, dtype=torch.int8),
            torch.empty(2, 1, dtype=torch.float16),
            -1,
            (1, 4),
        )


def test_dequantize_grouped_raise_error():
    with pytest.raises(ValueError):
        dequantize_grouped(
            torch.empty(2, 4, dtype=torch.int8),
            torch.empty(2, 1),
            torch.tensor((2,), dtype=torch.int32),
            -1,
            -1,
            (1, 4),
            torch.ones(2),
        )


def test_unpack_e2m1():
    codes = torch.zeros(2, 4, dtype=torch.uint8)
    values = unpack_e2m1(codes)

    assert values.dtype is torch.float32
    assert values.shape == (2, 8)
    assert values.is_contiguous()


def test_unpack_e2m1_raise_error():
    with pytest.raises(ValueError):
        unpack_e2m1(torch.empty(2, 2))


@pytest.mark.parametrize("dtype", NVFP4_DTYPES)
@pytest.mark.parametrize("contract_dim", NVFP4_CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", NVFP4_BLOCK_SHAPES)
@pytest.mark.parametrize("enable_global_scale", NVFP4_GLOBAL_SCALES)
@pytest.mark.parametrize("layout", NVFP4_LAYOUTS)
@pytest.mark.parametrize("shape", NVFP4_SHAPES)
def test_quantize_nvfp4(
    dtype, contract_dim, block_shape, enable_global_scale, layout, shape
):
    source = _make_nvfp4_source(shape, dtype, layout)
    original = source.clone()

    packed, scale, global_scale = quantize_nvfp4(
        source,
        contract_dim,
        block_shape,
        enable_global_scale,
        backend="eager",
    )

    expected_packed_shape = list(source.shape)
    expected_packed_shape[contract_dim] //= 2
    expected_scale_shape = list(source.shape)
    expected_scale_shape[contract_dim] //= 16
    assert packed.shape == tuple(expected_packed_shape)
    assert packed.dtype is torch.uint8
    assert packed.device == source.device
    assert packed.is_contiguous()
    assert scale.shape == tuple(expected_scale_shape)
    assert scale.dtype is torch.float8_e4m3fn
    assert scale.device == source.device
    assert scale.is_contiguous()
    if enable_global_scale:
        assert global_scale is not None
        assert global_scale.shape == ((1,) if source.ndim == 2 else source.shape[:1])
        assert global_scale.dtype is torch.float32
        assert global_scale.device == source.device
    else:
        assert global_scale is None
    assert torch.equal(source, original)


@pytest.mark.parametrize("output_layout", DENSE_OUTPUT_LAYOUTS)
@pytest.mark.parametrize("contract_dim", FP8_CONTRACT_DIMS)
@pytest.mark.parametrize("fmt", ("fp8", "int8", "mxfp8", "nvfp4"))
def test_quantize_dense_output_layout(output_layout, contract_dim, fmt):
    shape = (19, 32) if fmt == "nvfp4" else (19, 35)
    if contract_dim == -2:
        shape = shape[::-1]
    source = torch.arange(prod(shape), dtype=torch.float32).reshape(shape) / 17 - 20
    kwargs = {"backend": "eager", "output_layout": output_layout}
    if fmt == "fp8":
        codes, scale, _ = quantize_fp8(
            source, contract_dim, torch.float8_e4m3fn, (1, 16), **kwargs
        )
    elif fmt == "int8":
        codes, scale, _ = quantize_int8(source, contract_dim, (1, 16), **kwargs)
    elif fmt == "mxfp8":
        codes, scale, _ = quantize_mxfp8(source, contract_dim, (1, 16), **kwargs)
    else:
        codes, scale, _ = quantize_nvfp4(source, contract_dim, (1, 16), **kwargs)

    if output_layout == "column_major":
        assert codes.stride(-2) == 1
    else:
        assert codes.is_contiguous()
    assert scale.is_contiguous()


@pytest.mark.parametrize(
    ("shape", "dtype", "contract_dim", "block_shape"), NVFP4_ERROR_CASES
)
def test_quantize_nvfp4_raise_error(shape, dtype, contract_dim, block_shape):
    source = torch.empty(shape, dtype=dtype)
    with pytest.raises(ValueError):
        quantize_nvfp4(source, contract_dim, block_shape, backend="eager")


@pytest.mark.parametrize("block_shape", NVFP4_RECIPE_BLOCK_SHAPES)
@pytest.mark.parametrize("scale_dtype", NVFP4_RECIPE_SCALE_DTYPES)
@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize("enable_global_scale", (False, True))
def test_quantize_nvfp4_recipe(block_shape, scale_dtype, grouped, enable_global_scale):
    source = torch.zeros(8, 16)
    if grouped:
        result = quantize_nvfp4_grouped(
            source,
            torch.tensor((8, 8, 16), dtype=torch.int32),
            -1,
            -1,
            block_shape,
            backend="eager",
            enable_global_scale=enable_global_scale,
            scale_dtype=scale_dtype,
            qmax=4.0,
        )
    else:
        result = quantize_nvfp4(
            source,
            -1,
            block_shape,
            backend="eager",
            enable_global_scale=enable_global_scale,
            scale_dtype=scale_dtype,
            qmax=4.0,
        )
    codes, scales, global_scale = result

    assert codes.shape == (8, 8)
    assert codes.dtype is torch.uint8
    assert scales.dtype is scale_dtype
    if scale_dtype is torch.float8_e4m3fn and enable_global_scale:
        assert global_scale is not None
        assert global_scale.shape == ((3,) if grouped else (1,))
        return
    assert global_scale is None
    if scale_dtype is torch.float8_e4m3fn:
        return
    if grouped:
        without_global = quantize_nvfp4_grouped(
            source,
            torch.tensor((8, 8, 16), dtype=torch.int32),
            -1,
            -1,
            block_shape,
            backend="eager",
            enable_global_scale=False,
            scale_dtype=scale_dtype,
            qmax=4.0,
        )
    else:
        without_global = quantize_nvfp4(
            source,
            -1,
            block_shape,
            backend="eager",
            enable_global_scale=False,
            scale_dtype=scale_dtype,
            qmax=4.0,
        )
    assert without_global[2] is None
    assert torch.equal(codes, without_global[0])
    assert torch.equal(scales, without_global[1])


@pytest.mark.parametrize("dtype", MXFP8_DTYPES)
@pytest.mark.parametrize("fmt", MXFP8_FORMATS)
@pytest.mark.parametrize("contract_dim", MXFP8_CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", MXFP8_BLOCK_SHAPES)
@pytest.mark.parametrize("shape", MXFP8_SHAPES)
@pytest.mark.parametrize("layout", MXFP8_LAYOUTS)
@pytest.mark.parametrize("device", MXFP8_DEVICES)
@pytest.mark.parametrize("backend", MXFP8_BACKENDS)
@pytest.mark.parametrize("output_layout", MXFP8_OUTPUT_LAYOUTS)
def test_quantize_mxfp8(
    dtype, fmt, contract_dim, block_shape, shape, layout, device, backend, output_layout
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if backend == "cuda" and (
        device != "cuda" or torch.cuda.get_device_capability() < (8, 9)
    ):
        pytest.skip("CUDA MXFP8 requires SM89 or newer")
    source = torch.linspace(-448, 448, prod(shape), dtype=dtype, device=device).reshape(
        shape
    )
    if layout == "strided":
        source = source.repeat_interleave(2, dim=-1)[..., ::2]
    elif layout == "broadcast":
        source = source[:1].expand(*shape)
    original = source.clone()

    codes, scales, global_scale = quantize_mxfp8(
        source,
        contract_dim,
        block_shape,
        fmt,
        backend=backend,
        output_layout=output_layout,
    )

    expected_scale_shape = list(source.shape)
    expected_scale_shape[contract_dim] = (
        source.shape[contract_dim] + block_shape[1] - 1
    ) // block_shape[1]
    assert codes.shape == source.shape
    assert codes.dtype is str_to_dtype(fmt)
    assert (codes if output_layout == "row_major" else codes.mT).is_contiguous()
    assert scales.shape == tuple(expected_scale_shape)
    assert scales.dtype is torch.float8_e8m0fnu
    assert scales.is_contiguous()
    assert global_scale is None
    assert torch.equal(source, original)

    if block_shape[0] > 1:
        outer_dim = -1 if contract_dim == -2 else -2
        outer = scales.movedim(outer_dim, -2)
        for start in range(0, outer.shape[-2], block_shape[0]):
            count = min(block_shape[0], outer.shape[-2] - start)
            assert torch.equal(
                outer[..., start : start + count, :].view(torch.uint8),
                outer[..., start : start + 1, :]
                .expand(*outer.shape[:-2], count, -1)
                .view(torch.uint8),
            )


@pytest.mark.parametrize(("shape", "contract_dim"), MXFP8_SWIZZLE_EMPTY_CASES)
def test_quantize_mxfp8_scale_layout_empty(shape, contract_dim):
    codes, scales, global_scale = quantize_mxfp8(
        torch.empty(shape),
        contract_dim,
        (1, 32),
        backend="eager",
        scale_layout="swizzled_32_4_4",
    )

    assert codes.shape == shape
    assert scales.shape == (0,)
    assert global_scale is None


@pytest.mark.parametrize(
    ("shape", "contract_dim", "block_shape", "scale_layout"),
    (
        ((2, 32), -1, (1, 32), "invalid"),
        ((2, 32), -1, (1, 16), "swizzled_32_4_4"),
        ((2, 2, 32), -1, (1, 32), "swizzled_32_4_4"),
    ),
)
def test_quantize_mxfp8_scale_layout_raise_error(
    shape, contract_dim, block_shape, scale_layout
):
    with pytest.raises(ValueError):
        quantize_mxfp8(
            torch.empty(shape),
            contract_dim,
            block_shape,
            backend="eager",
            scale_layout=scale_layout,
        )


@pytest.mark.parametrize(
    (
        "shape",
        "contract_dim",
        "block_shape",
        "dtype",
        "fmt",
        "stochastic_rounding",
        "output_layout",
    ),
    MXFP8_ERROR_CASES,
)
def test_quantize_mxfp8_raise_error(
    shape, contract_dim, block_shape, dtype, fmt, stochastic_rounding, output_layout
):
    with pytest.raises(ValueError):
        quantize_mxfp8(
            torch.empty(shape, dtype=dtype),
            contract_dim,
            block_shape,
            fmt,
            stochastic_rounding,
            backend="eager",
            output_layout=output_layout,
        )


@pytest.mark.parametrize(("shape", "contract_dim"), MXFP8_EMPTY_CASES)
@pytest.mark.parametrize("block_shape", MXFP8_BLOCK_SHAPES)
@pytest.mark.parametrize("device", MXFP8_DEVICES)
@pytest.mark.parametrize("backend", MXFP8_BACKENDS)
@pytest.mark.parametrize("output_layout", MXFP8_OUTPUT_LAYOUTS)
def test_quantize_mxfp8_empty(
    shape, contract_dim, block_shape, device, backend, output_layout
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if backend == "cuda" and (
        device != "cuda" or torch.cuda.get_device_capability() < (8, 9)
    ):
        pytest.skip("CUDA MXFP8 requires SM89 or newer")
    source = torch.empty(shape, device=device)
    codes, scales, global_scale = quantize_mxfp8(
        source, contract_dim, block_shape, backend=backend, output_layout=output_layout
    )

    expected_scale_shape = list(source.shape)
    expected_scale_shape[contract_dim] = (
        source.shape[contract_dim] + block_shape[1] - 1
    ) // block_shape[1]
    assert codes.shape == source.shape
    assert codes.dtype is torch.float8_e4m3fn
    assert (codes if output_layout == "row_major" else codes.mT).is_contiguous()
    assert scales.shape == tuple(expected_scale_shape)
    assert scales.dtype is torch.float8_e8m0fnu
    assert scales.is_contiguous()
    assert global_scale is None


@pytest.mark.parametrize("block_shape", MXFP8_BLOCK_SHAPES)
def test_quantize_mxfp8_stochastic_rounding(block_shape):
    source = torch.full((130, 259), 1.0625)
    source[:, :: block_shape[1]] = 448

    torch.manual_seed(0)
    before_rne = torch.get_rng_state()
    rne_codes, rne_scales, _ = quantize_mxfp8(
        source, block_shape=block_shape, stochastic_rounding=False
    )
    assert torch.equal(torch.get_rng_state(), before_rne)

    torch.manual_seed(1)
    first_codes, first_scales, _ = quantize_mxfp8(
        source, block_shape=block_shape, stochastic_rounding=True
    )
    torch.manual_seed(1)
    second_codes, second_scales, _ = quantize_mxfp8(
        source, block_shape=block_shape, stochastic_rounding=True
    )

    assert torch.equal(first_codes.view(torch.uint8), second_codes.view(torch.uint8))
    assert torch.equal(first_scales.view(torch.uint8), second_scales.view(torch.uint8))
    assert torch.equal(rne_scales.view(torch.uint8), first_scales.view(torch.uint8))
    rounded = first_codes[source != 448].float()
    assert torch.all((rounded == 1.0) | (rounded == 1.125))
    # 5.1 times the worst-case binomial standard error.
    atol = 0.125 * 5.1 / (2 * rounded.numel() ** 0.5)
    torch.testing.assert_close(
        rounded.mean(), rounded.new_tensor(1.0625), rtol=0, atol=atol
    )
    assert rne_codes.shape == first_codes.shape


GROUPED_FAMILIES = ("fp8", "mxfp8", "nvfp4", "int8")
GROUPED_RAGGED_LAYOUTS = ("contract", "outer")
GROUPED_CONTRACT_DIMS = (-2, -1)
GROUPED_BLOCK_SHAPES = ((1, 16), (16, 16))


@pytest.mark.parametrize("family", GROUPED_FAMILIES)
@pytest.mark.parametrize("layout", GROUPED_RAGGED_LAYOUTS)
@pytest.mark.parametrize("contract_dim", GROUPED_CONTRACT_DIMS)
@pytest.mark.parametrize("block_shape", GROUPED_BLOCK_SHAPES)
def test_quantize_grouped(family, layout, contract_dim, block_shape):
    if layout == "contract":
        source = torch.linspace(-32, 32, 160).reshape(5, 32)
        ragged_dim = contract_dim
        offs = torch.tensor((16, 16, 32), dtype=torch.int32)
    else:
        source = torch.linspace(-32, 32, 288).reshape(9, 32)
        ragged_dim = -3 - contract_dim
        offs = torch.tensor((4, 4, 9), dtype=torch.int32)
    if contract_dim == -2:
        source = source.mT
    if family == "fp8":
        actual = quantize_fp8_grouped(
            source,
            offs,
            ragged_dim,
            contract_dim,
            torch.float8_e4m3fn,
            block_shape,
            backend="eager",
        )
    elif family == "mxfp8":
        actual = quantize_mxfp8_grouped(
            source, offs, ragged_dim, contract_dim, block_shape, backend="eager"
        )
    elif family == "nvfp4":
        actual = quantize_nvfp4_grouped(
            source,
            offs,
            ragged_dim,
            contract_dim,
            block_shape,
            backend="eager",
        )
    else:
        actual = quantize_int8_grouped(
            source, offs, ragged_dim, contract_dim, block_shape, 6, backend="eager"
        )
    expected_codes_shape = list(source.shape)
    if family == "nvfp4":
        expected_codes_shape[contract_dim] //= 2
    expected_scale_shape = list(source.shape)
    if layout == "contract":
        expected_scale_shape[contract_dim] = (
            source.shape[contract_dim] // block_shape[1] + offs.numel()
        )
    else:
        expected_scale_shape[contract_dim] = (
            source.shape[contract_dim] // block_shape[1]
        )
    assert actual[0].shape == tuple(expected_codes_shape)
    assert actual[1].shape == tuple(expected_scale_shape)
    assert actual[0].dtype is (
        torch.uint8
        if family == "nvfp4"
        else torch.int8
        if family == "int8"
        else torch.float8_e4m3fn
    )
    assert actual[1].dtype is (
        torch.float8_e4m3fn
        if family == "nvfp4"
        else torch.float8_e8m0fnu
        if family == "mxfp8"
        else torch.float32
    )
    if family == "nvfp4":
        assert actual[2] is not None and actual[2].shape == (offs.numel(),)
    else:
        assert actual[2] is None


GROUPED_ERROR_CASES = (
    ((2, 32), (16, 32), "int32", -1, 0, False),
    ((2, 32), (16, 32), "int32", 0, -1, False),
    ((2, 32), (16, 32), "float32", -1, -1, False),
    ((2, 32), (16, 32), "int32", -1, -1, True),
    ((2, 32), (8, 99, 16, 99, 32), "int32", -1, -1, False),
    ((2, 32, 1), (16, 32), "int32", -1, -1, False),
)


@pytest.mark.parametrize("case", GROUPED_ERROR_CASES)
def test_quantize_fp8_grouped_raise_error(case):
    shape, offset_values, dtype, ragged_dim, contract_dim, reshape = case
    offs = torch.tensor(offset_values, dtype=getattr(torch, dtype))
    if reshape:
        offs = offs.reshape(1, -1)
    elif len(offset_values) > 2:
        offs = offs[::2]
    with pytest.raises(ValueError):
        quantize_fp8_grouped(
            torch.empty(shape, device=offs.device),
            offs,
            ragged_dim,
            contract_dim,
            torch.float8_e4m3fn,
            (1, 16),
            backend="eager",
        )


@pytest.mark.parametrize("offs", ((15, 32), (16, 30)))
def test_quantize_nvfp4_grouped_raise_error(offs):
    offs = torch.tensor(offs, dtype=torch.int32)
    with pytest.raises(AssertionError):
        quantize_nvfp4_grouped(
            torch.empty(2, 32, device=offs.device),
            offs,
            -1,
            -1,
            (1, 16),
            backend="eager",
        )
