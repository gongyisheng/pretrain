import pytest
import torch
from src.model.components import MultiHeadAttention


def test_multihead_attention_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(2, 16, 64)
    out = mha(x)
    assert out.shape == (2, 16, 64)


def test_multihead_attention_is_causal():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    mha.eval()
    x = torch.randn(1, 8, 64)
    out_full = mha(x)
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = mha(x2)
    assert torch.allclose(out_full[0, :7], out_modified[0, :7], atol=1e-6)
