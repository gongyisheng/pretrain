import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from src.quant.int8 import quantize_int8, fake_quantize_int8, int8_gemm
from src.quant.linear import _gemm, QuantLinear
from src.quant.convert import apply_quantization
from src.utils.config import QuantConfig

int_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="int gemm kernel needs CUDA"
)

# qmax per bit width (all stored in torch.int8)
QMAX = {"int8": 127, "int7": 63, "int6": 31, "int5": 15, "int4": 7}
INT_FORMATS = ["int8", "int7", "int6", "int5", "int4"]


# --- quantize / fake-quantize (CPU-runnable) ---


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_int8_fake_quant_preserves_dtype_and_shape(dtype):
    x = torch.randn(8, 16, dtype=dtype)
    out = fake_quantize_int8(x, 127)
    assert out.dtype == dtype and out.shape == x.shape


@pytest.mark.parametrize("qmax", [127, 63, 31, 15, 7])
def test_int8s_quantize_symmetric_range_and_scale(qmax):
    torch.manual_seed(0)
    x = torch.randn(64, 64) * 10.0
    q, scale = quantize_int8(x, qmax)
    assert q.dtype == torch.int8  # all widths stored in int8
    assert scale.dtype == torch.float32 and scale.ndim == 0
    assert int(q.min()) >= -qmax and int(q.max()) <= qmax
    assert torch.isclose(scale, x.abs().max().float() / qmax)


def test_int8_quantize_rowwise_scale_shape():
    x = torch.randn(8, 16)
    _, scale = quantize_int8(x, 127, dim=-1)
    assert scale.shape == (8, 1)
    _, scale = quantize_int8(x, 127, dim=0)
    assert scale.shape == (1, 16)


@pytest.mark.parametrize("qmax", [127, 63, 31, 15, 7])
def test_int8s_fake_quant_recon_within_half_scale(qmax):
    torch.manual_seed(0)
    x = torch.randn(128, 128)
    q, scale = quantize_int8(x, qmax)
    recon = q.float() * scale
    assert (recon - x).abs().max() <= scale.item() / 2 + 1e-6


@pytest.mark.parametrize("qmax", [127, 7])
def test_int8s_zero_tensor_is_safe(qmax):
    recon = fake_quantize_int8(torch.zeros(4, 4), qmax)
    assert torch.isfinite(recon).all()


# --- shape-guard fallback (CPU-runnable: no torch._int_mm) ---


def test_int8_shape_guard_fallback_matches_oracle():
    torch.manual_seed(0)
    a = torch.randn(8, 128)  # M=8 <= 16 -> fallback
    b = torch.randn(128, 256)
    out = int8_gemm(a, b, torch.float32, 127, 127, rowwise=True)
    ref = fake_quantize_int8(a, 127, dim=-1) @ fake_quantize_int8(b, 127, dim=0)
    assert torch.allclose(out, ref.float(), atol=1e-4)


# --- int8_gemm real kernel (GPU-gated) ---


@int_gpu
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_int8_gemm_matches_bf16_reference(out_dtype):
    torch.manual_seed(0)
    M, K, N = 64, 128, 96
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    out = int8_gemm(a, b, out_dtype, 127, 127)
    ref = (a.float() @ b.float()).to(out_dtype)
    assert out.dtype == out_dtype and out.shape == (M, N)
    assert (out.float() - ref.float()).norm() / ref.float().norm() < 0.1


@int_gpu
def test_int8_gemm_matches_fake_quant_oracle():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    out = int8_gemm(a, b, torch.bfloat16, 127, 127)
    ref = fake_quantize_int8(a, 127) @ fake_quantize_int8(b, 127)
    assert (out.float() - ref.float()).norm() / ref.float().norm() < 0.02


@int_gpu
@pytest.mark.parametrize("qmax", [127, 63, 31, 15, 7])
def test_int8s_gemm_error_grows_as_bits_shrink(qmax):
    # fewer bits (smaller qmax) -> larger quantization error vs true matmul
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    ref = a.float() @ b.float()
    rel = (int8_gemm(a, b, torch.float32, qmax, qmax).float() - ref).norm() / ref.norm()
    tol = {127: 0.02, 63: 0.04, 31: 0.08, 15: 0.2, 7: 0.5}[qmax]
    assert rel < tol


@int_gpu
def test_int8s_gemm_mixed_w4a8():
    # weight int4 (qmax=7) x act int8 (qmax=127): both quantize to int8 storage
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    out = int8_gemm(a, b, torch.float32, 127, 7)
    ref = fake_quantize_int8(a, 127) @ fake_quantize_int8(b, 7)
    assert (out.float() - ref.float()).norm() / ref.float().norm() < 0.02


# --- _gemm dispatch ---


def test_gemm_passthrough_is_plain_matmul():
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    assert torch.allclose(_gemm(a, b, "bf16", "bf16", torch.float32), a @ b, atol=1e-4)


@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_int8s_gemm_mixed_family_uses_fake_quant(fmt):
    a, b = torch.randn(20, 32), torch.randn(32, 40)  # int x bf16 -> fallback
    out = _gemm(a, b, fmt, "bf16", torch.float32)
    assert out.shape == (20, 40) and torch.isfinite(out).all()


@int_gpu
@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_int8s_gemm_dispatches_to_kernel(fmt):
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda")
    b = torch.randn(128, 96, device="cuda")
    out = _gemm(a, b, fmt, fmt, torch.float32, rowwise=True)
    qmax = QMAX[fmt]
    ref = fake_quantize_int8(a, qmax, dim=-1) @ fake_quantize_int8(b, qmax, dim=0)
    assert (out - ref.float()).norm() / ref.float().norm() < 0.02


# --- config + converter ---


@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_int8s_recipe_expands(fmt):
    c = QuantConfig(enabled=True, dtype_recipe=fmt)
    assert c.dtype == {op: fmt for op in ("weight", "act", "input_grad", "weight_grad")}


@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_int8s_apply_quantization_swaps(fmt):
    model = nn.Sequential(nn.Linear(64, 64))
    rule = QuantConfig(enabled=True, dtype_recipe=fmt, include=["0"])
    cfg = SimpleNamespace(training=SimpleNamespace(quant=[rule]))
    apply_quantization(model, cfg)
    assert isinstance(model[0], QuantLinear)
