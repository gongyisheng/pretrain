import pytest
import torch

from src.quant.fp8 import quantize_fp8, fake_quantize_fp8, fp8_gemm

FP8_FORWARD_DTYPE = torch.float8_e4m3fn
FP8_GRAD_DTYPE = torch.float8_e5m2


def _fp8_capable():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="fp8 needs SM >= 8.9")


# --- quantize / fake-quantize (CPU-runnable) ---


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fake_quant_preserves_dtype_and_shape(dtype):
    x = torch.randn(8, 16, dtype=dtype)
    out = fake_quantize_fp8(x, FP8_FORWARD_DTYPE)
    assert out.dtype == dtype
    assert out.shape == x.shape


def test_quantize_scale_roundtrips_within_fp8_error():
    torch.manual_seed(0)
    x = torch.randn(64, 64) * 10.0
    xq, scale = quantize_fp8(x, FP8_FORWARD_DTYPE)
    assert xq.dtype == FP8_FORWARD_DTYPE
    assert scale.dtype == torch.float32 and scale.ndim == 0
    recon = xq.float() * scale
    rel = (recon - x).norm() / x.norm()
    assert rel < 0.1  # e4m3 per-tensor dynamic range


def test_zero_tensor_is_safe():
    x = torch.zeros(4, 4)
    recon = fake_quantize_fp8(x, FP8_FORWARD_DTYPE)
    assert torch.isfinite(recon).all()


# --- fp8_gemm on torch._scaled_mm (GPU-gated) ---


@fp8_only
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_fp8_gemm_matches_bf16_reference(out_dtype):
    torch.manual_seed(0)
    M, K, N = 64, 128, 96  # all divisible by 16
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    out = fp8_gemm(a, b, out_dtype, FP8_FORWARD_DTYPE, FP8_FORWARD_DTYPE)
    ref = (a.float() @ b.float()).to(out_dtype)
    assert out.dtype == out_dtype and out.shape == (M, N)
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.1


@fp8_only
def test_fp8_gemm_matches_fake_quant_oracle():
    # tighter check: same rounding on both sides
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    out = fp8_gemm(a, b, torch.bfloat16, FP8_FORWARD_DTYPE, FP8_FORWARD_DTYPE)
    ref = fake_quantize_fp8(a, FP8_FORWARD_DTYPE) @ fake_quantize_fp8(
        b, FP8_FORWARD_DTYPE
    )
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.02


@fp8_only
def test_fp8_gemm_rowwise_matches_reference():
    torch.manual_seed(0)
    M, K, N = 64, 128, 96
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    out = fp8_gemm(
        a, b, torch.bfloat16, FP8_FORWARD_DTYPE, FP8_FORWARD_DTYPE, rowwise=True
    )
    ref = a.float() @ b.float()
    rel = (out.float() - ref).norm() / ref.norm()
    assert out.shape == (M, N)
    assert rel < 0.06  # rowwise is tighter than tensorwise


@fp8_only
def test_fp8_gemm_e5m2_grad_side():
    # grad-output side is e5m2; the other side must be e4m3 (cuBLAS rule).
    torch.manual_seed(0)
    a = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 48, device="cuda", dtype=torch.bfloat16)
    out = fp8_gemm(a, b, torch.bfloat16, FP8_GRAD_DTYPE, FP8_FORWARD_DTYPE)
    ref = a.float() @ b.float()
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 0.15
