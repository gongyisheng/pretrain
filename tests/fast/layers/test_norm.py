"""Numerical parity tests: RMSNorm/LayerNorm vs eager fp32-accumulated reference."""

import pytest
import torch

from src.layers.norm import LayerNorm, RMSNorm
from tests.fast.layers._refs import SIMPLE_DTYPES, layernorm_ref, rmsnorm_ref


NORM_DTYPES = SIMPLE_DTYPES


# ---------------------------- LayerNorm ----------------------------


@pytest.mark.parametrize("shape", [(2, 16, 64), (8, 64), (64,)])
def test_layernorm_matches_ref_default(shape):
    d_model = shape[-1]
    layer = LayerNorm(d_model, eps=1e-5)
    x = torch.randn(*shape)
    assert torch.allclose(
        layer(x), layernorm_ref(x, layer.weight, layer.bias, layer.eps), atol=1e-5
    )


def test_layernorm_matches_ref_random_affine():
    d_model = 64
    layer = LayerNorm(d_model, eps=1e-5)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(d_model))
        layer.bias.copy_(torch.randn(d_model))
    x = torch.randn(2, 16, d_model)
    assert torch.allclose(
        layer(x), layernorm_ref(x, layer.weight, layer.bias, layer.eps), atol=1e-5
    )


def test_layernorm_no_bias():
    d_model = 64
    layer = LayerNorm(d_model, eps=1e-5, bias=False)
    assert layer.bias is None
    x = torch.randn(2, 16, d_model)
    assert torch.allclose(
        layer(x), layernorm_ref(x, layer.weight, None, layer.eps), atol=1e-5
    )


@pytest.mark.parametrize("dtype,atol", NORM_DTYPES)
def test_layernorm_dtype_parity(dtype, atol):
    d_model = 64
    layer = LayerNorm(d_model, eps=1e-5).to(dtype)
    x = torch.randn(2, 16, d_model, dtype=dtype)
    out = layer(x)
    assert out.dtype == dtype
    assert torch.allclose(
        out, layernorm_ref(x, layer.weight, layer.bias, layer.eps), atol=atol
    )


# ---------------------------- RMSNorm ----------------------------


@pytest.mark.parametrize("shape", [(2, 16, 64), (8, 64), (64,)])
def test_rmsnorm_matches_ref_default_weight(shape):
    d_model = shape[-1]
    layer = RMSNorm(d_model, eps=1e-6)
    x = torch.randn(*shape)
    assert torch.allclose(layer(x), rmsnorm_ref(x, layer.weight, layer.eps), atol=1e-5)


def test_rmsnorm_matches_ref_random_weight():
    d_model = 64
    layer = RMSNorm(d_model, eps=1e-6)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(d_model))
    x = torch.randn(2, 16, d_model)
    assert torch.allclose(layer(x), rmsnorm_ref(x, layer.weight, layer.eps), atol=1e-5)


@pytest.mark.parametrize("dtype,atol", NORM_DTYPES)
def test_rmsnorm_dtype_parity(dtype, atol):
    """Output preserves input dtype and matches eager ref within dtype tolerance."""
    d_model = 64
    layer = RMSNorm(d_model, eps=1e-6).to(dtype)
    x = torch.randn(2, 16, d_model, dtype=dtype)
    out = layer(x)
    assert out.dtype == dtype
    assert torch.allclose(out, rmsnorm_ref(x, layer.weight, layer.eps), atol=atol)


# ---------------------------- Numeric edge cases ----------------------------


@pytest.mark.parametrize(
    "dtype,atol", [(d, a) for d, a in NORM_DTYPES if d != torch.float32]
)
def test_rmsnorm_large_input_no_overflow(dtype, atol):
    """fp16: x^2 would overflow (>65504) without fp32 accumulation."""
    d_model = 64
    layer = RMSNorm(d_model, eps=1e-6).to(dtype)
    x = torch.full((2, 16, d_model), 1000.0, dtype=dtype)
    x += torch.randn_like(x) * 10
    out = layer(x)
    assert torch.isfinite(out).all(), "output has inf/nan"
    assert torch.allclose(out, rmsnorm_ref(x, layer.weight, layer.eps), atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_zero_input(dtype):
    """All-zeros input: rsqrt(0+eps) is large but x*rsqrt = 0; output must be finite zeros."""
    d_model = 64
    layer = RMSNorm(d_model, eps=1e-6).to(dtype)
    x = torch.zeros(2, 16, d_model, dtype=dtype)
    out = layer(x)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


@pytest.mark.parametrize(
    "dtype,atol", [(d, a) for d, a in NORM_DTYPES if d != torch.float32]
)
def test_layernorm_large_input_no_overflow(dtype, atol):
    """fp16: (x-mean)^2 with large magnitudes would overflow without fp32 reduction."""
    d_model = 64
    layer = LayerNorm(d_model, eps=1e-5).to(dtype)
    x = torch.full((2, 16, d_model), 1000.0, dtype=dtype)
    x += torch.randn_like(x) * 10
    out = layer(x)
    assert torch.isfinite(out).all(), "output has inf/nan"
    assert torch.allclose(
        out, layernorm_ref(x, layer.weight, layer.bias, layer.eps), atol=atol
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_layernorm_zero_input(dtype):
    """All-zeros input: var=0; output must equal bias (no nan from rsqrt(0+eps) * 0)."""
    d_model = 64
    layer = LayerNorm(d_model, eps=1e-5).to(dtype)
    with torch.no_grad():
        layer.bias.copy_(torch.randn(d_model, dtype=dtype))
    x = torch.zeros(2, 16, d_model, dtype=dtype)
    out = layer(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, layer.bias.expand_as(out), atol=1e-5)
