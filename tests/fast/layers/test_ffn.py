"""Numerical parity tests: SwiGluFFN and GeluFFN vs explicit formulas."""
import torch
import torch.nn.functional as F

from src.layers.ffn import GeluFFN, SwiGluFFN


# --- SwiGluFFN ---

def test_swiglu_ffn_matches_formula():
    """SwiGluFFN(x) == down_proj(silu(gate_proj(x)) * up_proj(x))."""
    d_model, intermediate_size = 64, 128
    ffn = SwiGluFFN(d_model, intermediate_size, dropout_ffn=0.0)
    ffn.eval()
    x = torch.randn(2, 16, d_model)

    gate = ffn.gate_proj(x)
    up = ffn.up_proj(x)
    ref = ffn.down_proj(F.silu(gate) * up)

    assert torch.allclose(ffn(x), ref, atol=1e-5)


def test_swiglu_ffn_output_shape():
    ffn = SwiGluFFN(d_model=64, intermediate_size=128, dropout_ffn=0.0)
    x = torch.randn(2, 16, 64)
    assert ffn(x).shape == (2, 16, 64)


# --- GeluFFN ---

def test_gelu_ffn_matches_formula():
    """GeluFFN(x) == fc2(gelu(fc1(x)))."""
    d_model, intermediate_size = 64, 256
    ffn = GeluFFN(d_model, intermediate_size, dropout_ffn=0.0)
    ffn.eval()
    x = torch.randn(2, 16, d_model)

    ref = ffn.fc2(F.gelu(ffn.fc1(x)))

    assert torch.allclose(ffn(x), ref, atol=1e-5)


def test_gelu_ffn_output_shape():
    ffn = GeluFFN(d_model=64, intermediate_size=256, dropout_ffn=0.0)
    x = torch.randn(2, 16, 64)
    assert ffn(x).shape == (2, 16, 64)
