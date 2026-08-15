"""Native cuBLASLt GEMM operator and adapter tests."""

import pytest
import torch

import src.kernel.backends.cublaslt._C  # noqa: F401
from src.kernel.backends.cublaslt import gemm as cublaslt_gemm


class _TensorMeta:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: str = "cuda:7",
        contiguous: bool = True,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.ndim = len(shape)
        self._contiguous = contiguous
        if strides is None:
            running = 1
            calculated = []
            for size in reversed(shape):
                calculated.append(running)
                running *= size
            self._strides = tuple(reversed(calculated))
        else:
            self._strides = strides

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self, dim: int | None = None):
        if dim is None:
            return self._strides
        return self._strides[dim]

    def element_size(self) -> int:
        return self.dtype.itemsize


def _scaled_mxfp8_metadata():
    return (
        _TensorMeta((129, 144), torch.float8_e4m3fn),
        _TensorMeta((144, 160), torch.float8_e4m3fn),
        _TensorMeta((129, 5), torch.float32),
        _TensorMeta((5, 160), torch.float32),
        torch.bfloat16,
        32,
        None,
        "fp8_e8m0",
    )


def test_native_operator_has_meta_implementation():
    a = torch.empty((129, 144), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty_strided(
        (144, 160), (1, 144), device="meta", dtype=torch.float8_e4m3fn
    )
    scale_a = torch.empty(2048, device="meta", dtype=torch.float8_e8m0fnu)
    scale_b = torch.empty(2048, device="meta", dtype=torch.float8_e8m0fnu)
    out = torch.ops.aot_kernel.scaled_gemm_mxfp8(a, b, scale_a, scale_b, None)
    assert out.shape == (129, 160)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "meta"


def test_swizzle_32_4_4_pads_and_maps_logical_scale_bytes():
    powers = torch.tensor(
        [2.0 ** (index % 10 - 5) for index in range(129 * 5)], dtype=torch.float32
    ).reshape(129, 5)

    actual = cublaslt_gemm._swizzle_32_4_4(powers)
    logical = powers.to(torch.float8_e8m0fnu).view(torch.uint8)
    actual_bytes = actual.view(torch.uint8)
    zero_byte = torch.tensor(0.0, dtype=torch.float8_e8m0fnu).view(torch.uint8)

    assert actual.dtype is torch.float8_e8m0fnu
    assert actual.is_contiguous()
    assert actual.numel() == 256 * 8
    for outer in range(128):
        for inner in range(4):
            offset = (outer % 32) * 16 + (outer // 32) * 4 + inner
            assert actual_bytes[offset].item() == logical[outer, inner].item()
    for row in range(256):
        for block in range(8):
            row_group, row_within_group = divmod(row, 128)
            row_tile, row_inner = divmod(row_within_group, 32)
            block_group, block_inner = divmod(block, 4)
            offset = (
                ((row_group * 2 + block_group) * 32 + row_inner) * 4 + row_tile
            ) * 4 + block_inner
            if row >= 129 or block >= 5:
                assert actual_bytes[offset].item() == zero_byte.item()


@pytest.mark.parametrize(
    ("name", "mutate", "expected_reason"),
    [
        (
            "e5m2_operands",
            lambda args: args.__setitem__(
                0, _TensorMeta((129, 144), torch.float8_e5m2)
            ),
            "operands",
        ),
        ("wrong_scale_dtype", lambda args: args.__setitem__(7, "fp32"), "scale_dtype"),
        ("wrong_block_size", lambda args: args.__setitem__(5, 64), "block_size"),
        (
            "invalid_logical_scale_shapes",
            lambda args: args.__setitem__(2, _TensorMeta((129, 4), torch.float32)),
            "scale shapes",
        ),
        ("fp16_output", lambda args: args.__setitem__(4, torch.float16), "bf16"),
        (
            "invalid_bias",
            lambda args: args.__setitem__(6, _TensorMeta((160,), torch.float16)),
            "bias dtype",
        ),
    ],
)
def test_can_implement_scaled_gemm_mxfp8_rejects_invalid_contracts(
    monkeypatch: pytest.MonkeyPatch, name: str, mutate, expected_reason: str
):
    monkeypatch.setattr(cublaslt_gemm, "_EXTENSION_ERROR", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_scaled_mxfp8_metadata())
    mutate(args)

    result = cublaslt_gemm.can_implement_scaled_gemm_mxfp8(tuple(args), {})

    assert not result.ok, name
    assert result.reason is not None
    assert expected_reason in result.reason


def test_can_implement_scaled_gemm_mxfp8_accepts_sm120_contract(monkeypatch):
    monkeypatch.setattr(cublaslt_gemm, "_EXTENSION_ERROR", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = cublaslt_gemm.can_implement_scaled_gemm_mxfp8(_scaled_mxfp8_metadata(), {})

    assert result.ok


def test_can_implement_scaled_gemm_mxfp8_rejects_missing_extension(monkeypatch):
    monkeypatch.setattr(cublaslt_gemm, "_EXTENSION_ERROR", "extension unavailable")

    result = cublaslt_gemm.can_implement_scaled_gemm_mxfp8(_scaled_mxfp8_metadata(), {})

    assert not result.ok
    assert result.reason == "extension unavailable"


def test_can_implement_scaled_gemm_mxfp8_rejects_sm90(monkeypatch):
    monkeypatch.setattr(cublaslt_gemm, "_EXTENSION_ERROR", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    result = cublaslt_gemm.can_implement_scaled_gemm_mxfp8(_scaled_mxfp8_metadata(), {})

    assert not result.ok
    assert result.reason is not None
    assert "SM100" in result.reason
