import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from src.quant.int8 import quantize_int8, fake_quantize_int8
from src.quant.linear import _gemm, QuantLinear
from src.quant.convert import apply_quantization
from src.quant.scale import fake_quantize_operand
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


# --- _gemm dispatch ---


def test_gemm_passthrough_is_plain_matmul():
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    assert torch.allclose(_gemm(a, b, "bf16", "bf16", torch.float32), a @ b, atol=1e-4)


@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_int8s_gemm_mixed_family_uses_fake_quant(fmt):
    a, b = torch.randn(20, 32), torch.randn(32, 40)  # int x bf16 -> fallback
    out = _gemm(a, b, fmt, "bf16", torch.float32)
    assert out.shape == (20, 40) and torch.isfinite(out).all()


_FMT_BY_QMAX = {v: k for k, v in QMAX.items()}


def _int_oracle(a, b, qmax, gran, bs):
    fmt = _FMT_BY_QMAX[qmax]
    return (
        fake_quantize_operand(a, -1, gran, bs, fmt).float()
        @ fake_quantize_operand(b, 0, gran, bs, fmt).float()
    )


@int_gpu
@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_int8s_gemm_dispatches_to_kernel(fmt):
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda")
    b = torch.randn(128, 96, device="cuda")
    out = _gemm(a, b, fmt, fmt, torch.float32, {"granularity": "rowwise"})
    ref = _int_oracle(a, b, QMAX[fmt], "rowwise", 128)
    assert (out - ref.float()).norm() / ref.float().norm() < 0.02


# --- config + converter ---


@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_int8s_recipe_expands(fmt):
    c = QuantConfig(enabled=True, dtype={"recipe": fmt})
    assert c.dtype == {op: fmt for op in ("weight", "act", "input_grad", "weight_grad")}


@pytest.mark.parametrize("fmt", INT_FORMATS)
def test_int8s_apply_quantization_swaps(fmt):
    model = nn.Sequential(nn.Linear(64, 64))
    rule = QuantConfig(enabled=True, dtype={"recipe": fmt}, include=["0"])
    cfg = SimpleNamespace(training=SimpleNamespace(quant=[rule]))
    apply_quantization(model, cfg)
    assert isinstance(model[0], QuantLinear)
