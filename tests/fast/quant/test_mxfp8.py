import pytest
import torch

from src.quant import mxfp8

MXFP8_FORWARD_DTYPE = torch.float8_e4m3fn
MXFP8_GRAD_DTYPE = torch.float8_e5m2

mxfp8_only = pytest.mark.skipif(
    not mxfp8.is_supported(), reason="mxfp8 needs CUDA compute capability >= 10.0"
)


# --- quantize / fake-quantize (CPU-runnable: the math is device-agnostic) ---


def test_scale_is_power_of_two_e8m0():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    xq, scale = mxfp8.quantize_mxfp8(x, MXFP8_FORWARD_DTYPE, dim=-1)
    assert xq.dtype == MXFP8_FORWARD_DTYPE
    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.shape == (4, 64 // mxfp8.MXFP8_BLOCK_SIZE)
    s = scale.float()
    log2s = torch.log2(s[s > 0])
    assert torch.allclose(log2s, log2s.round())  # exact powers of two


def test_fake_quant_matches_manual_dequant():
    torch.manual_seed(0)
    x = torch.randn(2, 64)
    xq, scale = mxfp8.quantize_mxfp8(x, MXFP8_FORWARD_DTYPE, dim=-1)
    manual = (xq.float().reshape(2, 2, 32) * scale.float().unsqueeze(-1)).reshape(2, 64)
    fq = mxfp8.fake_quantize_mxfp8(x, MXFP8_FORWARD_DTYPE, dim=-1)
    assert torch.allclose(fq, manual.to(fq.dtype), atol=1e-5)


def test_fake_quant_reasonable_error():
    torch.manual_seed(0)
    x = torch.randn(8, 128)
    fq = mxfp8.fake_quantize_mxfp8(x, MXFP8_FORWARD_DTYPE, dim=-1)
    assert (fq - x).norm() / x.norm() < 0.15


def test_quant_dim0_blocks_rows():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    xq, scale = mxfp8.quantize_mxfp8(x, MXFP8_FORWARD_DTYPE, dim=0)
    assert xq.shape == (64, 8)
    assert scale.shape == (64 // mxfp8.MXFP8_BLOCK_SIZE, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fake_quant_preserves_dtype(dtype):
    x = torch.randn(2, 64, dtype=dtype)
    assert mxfp8.fake_quantize_mxfp8(x, MXFP8_FORWARD_DTYPE, dim=-1).dtype == dtype


def test_zero_input_safe():
    x = torch.zeros(2, 64)
    fq = mxfp8.fake_quantize_mxfp8(x, MXFP8_FORWARD_DTYPE, dim=-1)
    assert torch.isfinite(fq).all()


def test_e5m2_element_also_quantizes():
    torch.manual_seed(0)
    x = torch.randn(2, 64)
    xq, scale = mxfp8.quantize_mxfp8(x, MXFP8_GRAD_DTYPE, dim=-1)
    assert xq.dtype == MXFP8_GRAD_DTYPE
    assert scale.dtype == torch.float8_e8m0fnu
    fq = mxfp8.fake_quantize_mxfp8(x, MXFP8_GRAD_DTYPE, dim=-1)
    assert torch.isfinite(fq).all()


def test_to_blocked_shape_and_dtype():
    # (rows, n_blocks) e8m0 scale -> flattened swizzled buffer, multiple of 128*4.
    scale = torch.randn(96, 5).to(torch.float8_e8m0fnu)
    blocked = mxfp8.to_blocked(scale)
    assert blocked.dtype == torch.float8_e8m0fnu
    # padded to 128 rows x 8 (=2*4) block-cols -> 128 * 8 = 1024 elements
    assert blocked.numel() == 128 * 8


# --- mxfp8_gemm on F.scaled_mm (GPU-gated: Blackwell tensor cores) ---


@mxfp8_only
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_mxfp8_gemm_matches_reference(out_dtype):
    torch.manual_seed(0)
    M, K, N = 256, 512, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    out = mxfp8.mxfp8_gemm(a, b, out_dtype, MXFP8_FORWARD_DTYPE, MXFP8_FORWARD_DTYPE)
    ref = a.float() @ b.float()
    assert out.dtype == out_dtype and out.shape == (M, N)
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 0.1


@mxfp8_only
def test_mxfp8_gemm_matches_fake_quant_oracle():
    torch.manual_seed(0)
    a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    out = mxfp8.mxfp8_gemm(
        a, b, torch.bfloat16, MXFP8_FORWARD_DTYPE, MXFP8_FORWARD_DTYPE
    )
    oracle = mxfp8.fake_quantize_mxfp8(
        a, MXFP8_FORWARD_DTYPE, -1
    ) @ mxfp8.fake_quantize_mxfp8(b, MXFP8_FORWARD_DTYPE, 0)
    rel = (out.float() - oracle.float()).norm() / oracle.float().norm()
    assert rel < 0.02


@mxfp8_only
def test_mxfp8_gemm_pads_unaligned_contraction():
    # K = 250 is not a multiple of 32; mxfp8_gemm zero-pads the contraction axis.
    torch.manual_seed(0)
    a = torch.randn(128, 250, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(250, 128, device="cuda", dtype=torch.bfloat16)
    out = mxfp8.mxfp8_gemm(
        a, b, torch.bfloat16, MXFP8_FORWARD_DTYPE, MXFP8_FORWARD_DTYPE
    )
    ref = a.float() @ b.float()
    assert out.shape == (128, 128)
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 0.1


@mxfp8_only
def test_mxfp8_gemm_requires_e4m3():
    # The BlockWise1x32 MMA only accepts e4m3 x e4m3; e5m2 is rejected. mxfp8
    # therefore runs all three GEMMs in e4m3 (see QUANT_DTYPE_RECIPES["mxfp8"]).
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(Exception):
        mxfp8.mxfp8_gemm(a, b, torch.bfloat16, MXFP8_GRAD_DTYPE, MXFP8_FORWARD_DTYPE)
