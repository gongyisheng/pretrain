import pytest
import torch

from src.quant.fp8 import quantize_fp8, fake_quantize_fp8

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
