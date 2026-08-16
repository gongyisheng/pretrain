"""Native cuBLASLt GEMM operator and adapter tests."""

import pytest
import torch

import src.kernel.backends.cublaslt._C  # noqa: F401
from src.kernel.backends.cublaslt import gemm as cublaslt_gemm
from src.kernel.ops.gemm import scaled_gemm
from src.quant.quantize import quantize_operand


_MXFP8_SCALING = {
    "granularity": "blockwise",
    "block_shape": (1, 32),
    "scale_dtype": "fp8_e8m0",
}

cuda_mxfp8_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuBLASLt MXFP8 GEMM requires CUDA"
)


def _require_mxfp8_device() -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuBLASLt MXFP8 GEMM requires SM100 or newer")


def _mxfp8_operands(
    shape: tuple[int, int, int], scale_case: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K, N = shape
    blocks = (K + 31) // 32
    if scale_case == "unit":
        block = torch.linspace(-256.0, 256.0, 32, device="cuda", dtype=torch.bfloat16)
        a = block.repeat(M, blocks)[:, :K].contiguous()
        b = block[:, None].repeat(blocks, N)[:K].contiguous()
    else:
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    aq, scale_a = quantize_operand(a, -1, "fp8_e4m3", _MXFP8_SCALING)
    bq, scale_b = quantize_operand(b, -2, "fp8_e4m3", _MXFP8_SCALING)
    if scale_case == "unit":
        assert torch.equal(scale_a, torch.ones_like(scale_a))
        assert torch.equal(scale_b, torch.ones_like(scale_b))
    else:
        assert scale_a.unique().numel() > 1
        assert scale_b.unique().numel() > 1
    return aq, bq, scale_a, scale_b


def _assert_matches_eager(
    aq: torch.Tensor,
    bq: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    eager = scaled_gemm(
        aq,
        bq,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        bias=bias,
        scale_dtype="fp8_e8m0",
        backend="eager",
    )
    actual = scaled_gemm(
        aq,
        bq,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        bias=bias,
        scale_dtype="fp8_e8m0",
        backend="cublaslt",
    )
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert actual.stride() == (actual.shape[1], 1)
    difference = actual.float() - eager.float()
    relative_error = (difference.norm() / eager.float().norm()).item()
    maximum_error = difference.abs().max().item()
    assert torch.equal(actual, eager), (relative_error, maximum_error)


@cuda_mxfp8_only
@pytest.mark.parametrize("shape", [(128, 128, 128), (129, 144, 160), (256, 384, 512)])
@pytest.mark.parametrize("scale_case", ["unit", "varying"])
def test_cublaslt_mxfp8_precision_matches_eager_bitwise(
    shape: tuple[int, int, int], scale_case: str
):
    _require_mxfp8_device()
    _assert_matches_eager(*_mxfp8_operands(shape, scale_case))


@cuda_mxfp8_only
def test_cublaslt_mxfp8_bias_matches_eager_bitwise():
    _require_mxfp8_device()
    aq, bq, scale_a, scale_b = _mxfp8_operands((129, 144, 160), "varying")
    bias = torch.linspace(-128.0, 128.0, 160, device="cuda", dtype=torch.bfloat16)
    without_bias = scaled_gemm(
        aq,
        bq,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        scale_dtype="fp8_e8m0",
        backend="eager",
    )
    with_bias = scaled_gemm(
        aq,
        bq,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        bias=bias,
        scale_dtype="fp8_e8m0",
        backend="eager",
    )
    assert not torch.equal(with_bias, without_bias)
    _assert_matches_eager(aq, bq, scale_a, scale_b, bias)


def test_cublaslt_adapter_has_meta_implementation():
    a = torch.empty((129, 144), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty((144, 160), device="meta", dtype=torch.float8_e4m3fn)
    scale_a = torch.empty((129, 5), device="meta")
    scale_b = torch.empty((5, 160), device="meta")
    out = cublaslt_gemm.scaled_gemm_mxfp8(
        a,
        b,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        scale_dtype=torch.float8_e8m0fnu,
    )
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
