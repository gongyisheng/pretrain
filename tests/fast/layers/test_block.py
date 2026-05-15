import torch
import torch.nn as nn

from src.layers.attention import MultiHeadAttention
from src.layers.block import BaseTransformerBlock
from src.layers.ffn import FFN


class _MinimalTransformerBlock(BaseTransformerBlock):
    """Minimal concrete subclass of BaseTransformerBlock used in tests."""

    def __init__(self, d_model: int, n_heads: int, intermediate_size: int, dropout_attn: float = 0.0, dropout_ffn: float = 0.0):
        super().__init__(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout_attn)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, intermediate_size, activation="gelu", dropout=dropout_ffn)

    def attn_sublayer(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return self.attn(self.ln1(x))

    def ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ln2(x))


def test_transformer_block_output_shape():
    block = _MinimalTransformerBlock(d_model=64, n_heads=4, intermediate_size=256, dropout_attn=0.0)
    x = torch.randn(2, 16, 64)
    out, _ = block(x)
    assert out.shape == (2, 16, 64)


def test_transformer_block_residual():
    block = _MinimalTransformerBlock(d_model=64, n_heads=4, intermediate_size=256, dropout_attn=0.0)
    x = torch.randn(2, 16, 64)
    out, _ = block(x)
    assert not torch.allclose(out, x)
