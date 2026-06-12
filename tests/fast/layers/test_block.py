import pytest
import torch
import torch.nn as nn

from src.layers.attention import MultiHeadAttention
from src.layers.block import BaseTransformerBlock
from src.layers.mlp import DenseMLPBlock
from tests.fast.helpers import ATTN_IMPLEMENTATION, make_attn_mask, skip_if_unsupported


class _MinimalTransformerBlock(BaseTransformerBlock):
    """Minimal concrete subclass of BaseTransformerBlock used in tests."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        intermediate_size: int,
        dropout_attn: float = 0.0,
        dropout_ffn: float = 0.0,
        attn_implementation: str = "sdpa",
    ):
        super().__init__(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model, n_heads, dropout_attn, attn_implementation=attn_implementation
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = DenseMLPBlock(
            d_model,
            intermediate_size=intermediate_size,
            activation="gelu",
            gated=False,
            dropout=dropout_ffn,
        )

    def attn_sublayer(self, x: torch.Tensor, attn_mask=None, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return self.attn(self.ln1(x), attn_mask=attn_mask)

    def mlp_sublayer(self, x: torch.Tensor) -> tuple:
        return self.mlp(self.ln2(x))


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_transformer_block_output_shape(impl, device):
    skip_if_unsupported(impl, device)
    block = _MinimalTransformerBlock(
        d_model=64,
        n_heads=4,
        intermediate_size=256,
        dropout_attn=0.0,
        attn_implementation=impl,
    )
    x = torch.randn(2, 16, 64)
    pos = torch.arange(16).unsqueeze(0).expand(2, -1)
    attn_mask, _ = make_attn_mask("causal", impl, pos, x.dtype)
    out, _, aux = block(x, attn_mask=attn_mask)
    assert out.shape == (2, 16, 64)
    assert aux is None


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_transformer_block_residual(impl, device):
    skip_if_unsupported(impl, device)
    block = _MinimalTransformerBlock(
        d_model=64,
        n_heads=4,
        intermediate_size=256,
        dropout_attn=0.0,
        attn_implementation=impl,
    )
    x = torch.randn(2, 16, 64)
    pos = torch.arange(16).unsqueeze(0).expand(2, -1)
    attn_mask, _ = make_attn_mask("causal", impl, pos, x.dtype)
    out, _, _ = block(x, attn_mask=attn_mask)
    assert not torch.allclose(out, x)
