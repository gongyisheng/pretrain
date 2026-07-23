import pytest
import torch

from src.layers.block import TransformerBlock
from src.utils.config import ModelConfig
from tests.fast.helpers import ATTN_IMPLEMENTATION, make_attn_mask, skip_if_unsupported


def _block_cfg(impl):
    return ModelConfig(
        d_model=64,
        n_layers=1,
        vocab_size=256,
        attn=[
            {
                "attn_cls": "mha",
                "attn_kwargs": {
                    "n_heads": 4,
                    "bias": True,
                    "attn_implementation": impl,
                },
            }
        ],
        mlp=[
            {
                "mlp_cls": "dense",
                "mlp_kwargs": {"activation": "gelu", "gated": False, "bias": True},
            }
        ],
        norm_cls="layernorm",
        pos_emb_cls="learned",
        pos_emb_kwargs={},
    )


def _run(impl):
    block = TransformerBlock(_block_cfg(impl), layer_idx=0)
    x = torch.randn(2, 16, 64)
    pos = torch.arange(16).unsqueeze(0).expand(2, -1)
    attn_mask, _ = make_attn_mask("causal", impl, pos, x.dtype)
    return x, block(x, attn_mask=attn_mask)


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_transformer_block_output_shape(impl, device):
    skip_if_unsupported(impl, device)
    _, (out, _, aux) = _run(impl)
    assert out.shape == (2, 16, 64)
    assert aux is None


@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_transformer_block_residual(impl, device):
    skip_if_unsupported(impl, device)
    x, (out, _, _) = _run(impl)
    assert not torch.allclose(out, x)
