"""Numerical parity tests for the unified FFN module.

Covers both the ungated path (up_proj → act → down_proj) and the gated path
(down_proj ∘ act(gate_proj(x), up_proj(x))) across activations.
"""
import pytest
import torch
import torch.nn.functional as F

from src.layers.ffn import FFN


# --- Ungated FFN ---

def test_ffn_gelu_ungated_matches_formula():
    """FFN(activation=gelu, gated=False) = down_proj(gelu(up_proj(x)))."""
    ffn = FFN(d_model=64, intermediate_size=256, activation="gelu", gated=False, dropout=0.0)
    ffn.eval()
    x = torch.randn(2, 16, 64)
    ref = ffn.down_proj(F.gelu(ffn.up_proj(x)))
    assert torch.allclose(ffn(x), ref, atol=1e-5)


def test_ffn_relu_ungated_matches_formula():
    ffn = FFN(d_model=64, intermediate_size=256, activation="relu", gated=False, dropout=0.0)
    ffn.eval()
    x = torch.randn(2, 16, 64)
    ref = ffn.down_proj(F.relu(ffn.up_proj(x)))
    assert torch.allclose(ffn(x), ref, atol=1e-5)


def test_ffn_silu_ungated_matches_formula():
    ffn = FFN(d_model=64, intermediate_size=256, activation="silu", gated=False, dropout=0.0)
    ffn.eval()
    x = torch.randn(2, 16, 64)
    ref = ffn.down_proj(F.silu(ffn.up_proj(x)))
    assert torch.allclose(ffn(x), ref, atol=1e-5)


# --- Gated FFN (GLU family) ---

def test_ffn_silu_glu_matches_formula():
    """gated=True, activation=silu (literature: SwiGLU)."""
    ffn = FFN(d_model=64, intermediate_size=128, activation="silu", gated=True, dropout=0.0)
    ffn.eval()
    x = torch.randn(2, 16, 64)
    ref = ffn.down_proj(F.silu(ffn.gate_proj(x)) * ffn.up_proj(x))
    assert torch.allclose(ffn(x), ref, atol=1e-5)


def test_ffn_gelu_glu_matches_formula():
    """gated=True, activation=gelu (literature: GeGLU)."""
    ffn = FFN(d_model=64, intermediate_size=128, activation="gelu", gated=True, dropout=0.0)
    ffn.eval()
    x = torch.randn(2, 16, 64)
    ref = ffn.down_proj(F.gelu(ffn.gate_proj(x)) * ffn.up_proj(x))
    assert torch.allclose(ffn(x), ref, atol=1e-5)


def test_ffn_relu_glu_matches_formula():
    """gated=True, activation=relu (literature: ReGLU)."""
    ffn = FFN(d_model=64, intermediate_size=128, activation="relu", gated=True, dropout=0.0)
    ffn.eval()
    x = torch.randn(2, 16, 64)
    ref = ffn.down_proj(F.relu(ffn.gate_proj(x)) * ffn.up_proj(x))
    assert torch.allclose(ffn(x), ref, atol=1e-5)


# --- Output shapes ---

@pytest.mark.parametrize("gated", [False, True])
def test_ffn_output_shape(gated):
    ffn = FFN(d_model=64, intermediate_size=128, activation="silu", gated=gated, dropout=0.0)
    x = torch.randn(2, 16, 64)
    assert ffn(x).shape == (2, 16, 64)


# --- Bias ---

@pytest.mark.parametrize("gated", [False, True])
def test_ffn_default_bias_is_false(gated):
    """FFN defaults to bias=False (Qwen3 / Llama convention)."""
    ffn = FFN(d_model=64, intermediate_size=128, activation="silu", gated=gated)
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None
    if gated:
        assert ffn.gate_proj.bias is None


@pytest.mark.parametrize("gated", [False, True])
def test_ffn_bias_true_adds_biases(gated):
    """Explicit bias=True adds biases to all projections."""
    ffn = FFN(d_model=64, intermediate_size=128, activation="silu", gated=gated, bias=True)
    assert ffn.up_proj.bias is not None
    assert ffn.down_proj.bias is not None
    if gated:
        assert ffn.gate_proj.bias is not None


# --- Error handling ---

def test_ffn_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        FFN(d_model=64, intermediate_size=128, activation="mish")


def test_ffn_unknown_activation_in_gated_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        FFN(d_model=64, intermediate_size=128, activation="mish", gated=True)
