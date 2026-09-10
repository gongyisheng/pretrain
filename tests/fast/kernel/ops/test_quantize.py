"""Packed E2M1 conversion contracts."""

from math import prod

import pytest
import torch

from src.kernel.ops import pack_e2m1_rne


DTYPES = (torch.float32, torch.float16, torch.bfloat16)
DIMS = (-2, -1)
DEVICES = ("cpu", "cuda")
BACKENDS = (None, "eager", "triton")
LAYOUTS = ("dense", "strided", "broadcast")
SHAPES = ((2, 8), (2, 0))
ERROR_CASES = (
    ((2, 4), 0, torch.float32),
    ((2, 4), -3, torch.float32),
    ((2, 3), -1, torch.float32),
    ((3, 2), -2, torch.float32),
    ((), -1, torch.float32),
    ((4,), -2, torch.float32),
    ((2, 4), -1, torch.int32),
)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", SHAPES)
def test_pack_e2m1_rne(dtype, dim, device, backend, layout, shape):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if backend == "triton" and (
        device != "cuda" or torch.cuda.get_device_capability()[0] not in (10, 11, 12)
    ):
        pytest.skip("Triton E2M1 conversion requires Blackwell")

    source = torch.arange(prod(shape), dtype=dtype, device=device).reshape(shape)
    if layout == "strided":
        source = source.repeat_interleave(2, dim=-1)[..., ::2]
    elif layout == "broadcast":
        source = source[:1].expand(2, -1)
    original = source.clone()

    actual = pack_e2m1_rne(source, dim, backend)

    assert actual.dtype is torch.uint8
    assert actual.device == source.device
    assert actual.is_contiguous()
    expected_shape = list(source.shape)
    expected_shape[dim] //= 2
    assert actual.shape == tuple(expected_shape)
    storage_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(source.view(storage_dtype), original.view(storage_dtype))


@pytest.mark.parametrize(("shape", "dim", "dtype"), ERROR_CASES)
def test_pack_e2m1_rne_raise_error(shape, dim, dtype):
    with pytest.raises(ValueError):
        pack_e2m1_rne(torch.empty(shape, dtype=dtype, device="cpu"), dim)
