import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from src.quant.int8 import quantize_int8, fake_quantize_int8, int8_gemm
from src.quant.linear import _gemm, QuantLinear
from src.quant.convert import apply_quantization
from src.utils.config import QuantConfig

int8_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="int8 gemm kernel needs CUDA"
)


# --- quantize / fake-quantize (CPU-runnable) ---


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fake_quant_preserves_dtype_and_shape(dtype):
    x = torch.randn(8, 16, dtype=dtype)
    out = fake_quantize_int8(x)
    assert out.dtype == dtype and out.shape == x.shape


def test_quantize_symmetric_range_and_scale():
    torch.manual_seed(0)
    x = torch.randn(64, 64) * 10.0
    q, scale = quantize_int8(x)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float32 and scale.ndim == 0
    assert int(q.min()) >= -127 and int(q.max()) <= 127  # symmetric, no -128
    assert torch.isclose(scale, x.abs().max().float() / 127)


def test_quantize_rowwise_scale_shape():
    x = torch.randn(8, 16)
    _, scale = quantize_int8(x, dim=-1)
    assert scale.shape == (8, 1)
    _, scale = quantize_int8(x, dim=0)
    assert scale.shape == (1, 16)


def test_fake_quant_recon_within_half_scale():
    torch.manual_seed(0)
    x = torch.randn(128, 128)
    q, scale = quantize_int8(x)
    recon = q.float() * scale
    assert (recon - x).abs().max() <= scale.item() / 2 + 1e-6


def test_zero_tensor_is_safe():
    recon = fake_quantize_int8(torch.zeros(4, 4))
    assert torch.isfinite(recon).all()


# --- shape-guard fallback (CPU-runnable: no torch._int_mm) ---


def test_shape_guard_fallback_matches_oracle():
    torch.manual_seed(0)
    a = torch.randn(8, 128)  # M=8 <= 16 -> fallback
    b = torch.randn(128, 256)
    out = int8_gemm(a, b, torch.float32, rowwise=True)
    ref = fake_quantize_int8(a, dim=-1) @ fake_quantize_int8(b, dim=0)
    assert torch.allclose(out, ref.float(), atol=1e-4)


# --- int8_gemm real kernel (GPU-gated) ---


@int8_gpu
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_int8_gemm_matches_bf16_reference(out_dtype):
    torch.manual_seed(0)
    M, K, N = 64, 128, 96
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    out = int8_gemm(a, b, out_dtype)
    ref = (a.float() @ b.float()).to(out_dtype)
    assert out.dtype == out_dtype and out.shape == (M, N)
    assert (out.float() - ref.float()).norm() / ref.float().norm() < 0.1


@int8_gpu
def test_int8_gemm_matches_fake_quant_oracle():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    out = int8_gemm(a, b, torch.bfloat16)
    ref = fake_quantize_int8(a) @ fake_quantize_int8(b)
    assert (out.float() - ref.float()).norm() / ref.float().norm() < 0.02


@int8_gpu
def test_int8_gemm_rowwise_tighter_than_tensorwise():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    ref = a.float() @ b.float()
    tw = (int8_gemm(a, b, torch.float32).float() - ref).norm() / ref.norm()
    rw = (int8_gemm(a, b, torch.float32, rowwise=True).float() - ref).norm() / ref.norm()
    assert rw <= tw


# --- _gemm dispatch ---


def test_gemm_passthrough_is_plain_matmul():
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    assert torch.allclose(_gemm(a, b, "bf16", "bf16", torch.float32), a @ b, atol=1e-4)


def test_gemm_mixed_family_uses_fake_quant():
    a, b = torch.randn(20, 32), torch.randn(32, 40)  # int8 x bf16 -> fallback
    out = _gemm(a, b, "int8", "bf16", torch.float32)
    assert out.shape == (20, 40) and torch.isfinite(out).all()


@int8_gpu
def test_gemm_int8_dispatches_to_kernel():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda")
    b = torch.randn(128, 96, device="cuda")
    out = _gemm(a, b, "int8", "int8", torch.float32, rowwise=True)
    ref = fake_quantize_int8(a, dim=-1) @ fake_quantize_int8(b, dim=0)
    assert (out - ref.float()).norm() / ref.float().norm() < 0.02


# --- config + converter ---


def test_int8_recipe_expands():
    c = QuantConfig(enabled=True, dtype_recipe="int8")
    assert c.dtype == {
        "weight": "int8",
        "act": "int8",
        "input_grad": "int8",
        "weight_grad": "int8",
    }


def test_apply_quantization_swaps_int8():
    model = nn.Sequential(nn.Linear(64, 64))
    rule = QuantConfig(enabled=True, dtype_recipe="int8", include=["0"])
    cfg = SimpleNamespace(training=SimpleNamespace(quant=[rule]))
    apply_quantization(model, cfg)
    assert isinstance(model[0], QuantLinear)
