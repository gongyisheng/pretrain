"""Numerical parity tests: RMSNorm vs torch.nn.RMSNorm."""
import pytest
import torch
import torch.nn as nn

from src.layers.norm import RMSNorm


@pytest.mark.parametrize("shape", [(2, 16, 64), (8, 64), (64,)])
def test_rmsnorm_matches_nn_rmsnorm_default_weight(shape):
    """Our RMSNorm equals nn.RMSNorm when both use the default ones-weight."""
    d_model = shape[-1]
    ours = RMSNorm(d_model, eps=1e-6)
    ref = nn.RMSNorm(d_model, eps=1e-6, elementwise_affine=True)

    x = torch.randn(*shape)
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


def test_rmsnorm_random_weight_parity():
    """Parity holds when both modules share a non-trivial weight vector."""
    d_model = 64
    ours = RMSNorm(d_model, eps=1e-6)
    ref = nn.RMSNorm(d_model, eps=1e-6, elementwise_affine=True)
    with torch.no_grad():
        w = torch.randn(d_model)
        ours.weight.copy_(w)
        ref.weight.copy_(w)

    x = torch.randn(2, 16, d_model)
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


@pytest.mark.parametrize(
    "dtype,atol",
    [
        (torch.float32, 1e-5),
        (torch.float16, 1e-2),
        (torch.bfloat16, 2e-2),
    ],
)
def test_rmsnorm_dtype_preservation(dtype, atol):
    """Output preserves input dtype and matches nn.RMSNorm within dtype tolerance."""
    d_model = 64
    ours = RMSNorm(d_model, eps=1e-6).to(dtype)
    ref = nn.RMSNorm(d_model, eps=1e-6, elementwise_affine=True).to(dtype)
    x = torch.randn(2, 16, d_model, dtype=dtype)

    out_ours = ours(x)
    out_ref = ref(x)

    assert out_ours.dtype == dtype
    assert torch.allclose(out_ours, out_ref, atol=atol)
