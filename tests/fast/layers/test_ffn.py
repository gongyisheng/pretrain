"""FFN tests: behavior (shape, bias, error cases) + numerical parity vs ffn_ref."""

import pytest
import torch

from src.layers.activation import UNGATED_ACTIVATIONS
from src.layers.ffn import FFN
from tests.fast.layers._refs import SIMPLE_DTYPES, ffn_ref


ACT_NAMES = list(UNGATED_ACTIVATIONS.keys())


# --- Behavior ---


@pytest.mark.parametrize("gated", [False, True])
def test_ffn_output_shape(gated):
    ffn = FFN(
        d_model=64, intermediate_size=128, activation="silu", gated=gated, dropout=0.0
    )
    x = torch.randn(2, 16, 64)
    assert ffn(x).shape == (2, 16, 64)


@pytest.mark.parametrize("gated", [False, True])
def test_ffn_default_bias_is_false(gated):
    """FFN defaults to bias=False (Qwen3 / Llama convention)."""
    ffn = FFN(d_model=64, intermediate_size=128, activation="silu", gated=gated)
    w1 = ffn.gate_up_proj if gated else ffn.up_proj
    assert w1.bias is None
    assert ffn.down_proj.bias is None


@pytest.mark.parametrize("gated", [False, True])
def test_ffn_bias_true_adds_biases(gated):
    """Explicit bias=True adds biases to all projections."""
    ffn = FFN(
        d_model=64, intermediate_size=128, activation="silu", gated=gated, bias=True
    )
    w1 = ffn.gate_up_proj if gated else ffn.up_proj
    assert w1.bias is not None
    assert ffn.down_proj.bias is not None


def test_ffn_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        FFN(d_model=64, intermediate_size=128, activation="mish")


def test_ffn_unknown_activation_in_gated_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        FFN(d_model=64, intermediate_size=128, activation="mish", gated=True)


# --- Numerical parity ---


@pytest.mark.parametrize("activation", ACT_NAMES)
@pytest.mark.parametrize("dtype,atol", SIMPLE_DTYPES)
def test_ffn_ungated_matches_ref(activation, dtype, atol):
    """FFN(gated=False, activation=A) = down_proj(A(up_proj(x)))."""
    torch.manual_seed(0)
    ffn = FFN(
        d_model=64,
        intermediate_size=256,
        activation=activation,
        gated=False,
        dropout=0.0,
    ).to(dtype)
    ffn.eval()
    x = torch.randn(2, 16, 64, dtype=dtype)
    out = ffn(x)
    out_ref = ffn_ref(x, ffn.down_proj, activation=activation, up_proj=ffn.up_proj)
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("activation", ACT_NAMES)
@pytest.mark.parametrize("dtype,atol", SIMPLE_DTYPES)
def test_ffn_gated_matches_ref(activation, dtype, atol):
    """FFN(gated=True, activation=A) = down_proj(A(gate, up)) with fused gate_up_proj."""
    torch.manual_seed(0)
    ffn = FFN(
        d_model=64,
        intermediate_size=128,
        activation=activation,
        gated=True,
        dropout=0.0,
    ).to(dtype)
    ffn.eval()
    x = torch.randn(2, 16, 64, dtype=dtype)
    out = ffn(x)
    out_ref = ffn_ref(
        x, ffn.down_proj, activation=activation, gate_up_proj=ffn.gate_up_proj
    )
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)
