import torch
import torch.nn as nn

from src.layers.attention import MultiHeadAttention
from src.layers.block import BaseTransformerBlock
from src.layers.ffn import FFN
from src.layers.norm import LayerNorm
from src.layers.pos_emb import LearnedPositionalEmbedding
from src.layers.residual import RESIDUAL_REGISTRY
from src.utils.config import ModelConfig


class GPT2TransformerBlock(BaseTransformerBlock):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        intermediate_size: int,
        dropout_attn: float,
        dropout_ffn: float,
        qk_norm: bool = False,
        attn_bias: bool = True,
        mlp_bias: bool = True,
        mlp_activation: str = "gelu",
        mlp_gated: bool = False,
        attn_implementation: str = "flex_attention",
        **kwargs,
    ):
        super().__init__(d_model, **kwargs)
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model,
            n_heads,
            dropout_attn,
            qk_norm=qk_norm,
            bias=attn_bias,
            attn_implementation=attn_implementation,
        )
        self.ln2 = LayerNorm(d_model)
        self.ffn = FFN(
            d_model,
            intermediate_size,
            activation=mlp_activation,
            gated=mlp_gated,
            bias=mlp_bias,
            dropout=dropout_ffn,
        )

    def attn_sublayer(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        return self.attn(self.ln1(x), attn_mask=attn_mask)

    def ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ln2(x))


class GPT2Model(nn.Module):
    def __init__(self, config: ModelConfig, max_seq_len: int = 1024):
        super().__init__()
        self.config = config

        # Pad vocab to multiple of 128 for better matmul alignment
        pad_multiple = 128
        self.padded_vocab_size = (
            (config.vocab_size + pad_multiple - 1) // pad_multiple
        ) * pad_multiple
        self.token_emb = nn.Embedding(self.padded_vocab_size, config.d_model)
        self.pos_emb = LearnedPositionalEmbedding(max_seq_len, config.d_model)
        self.dropout_emb = nn.Dropout(config.dropout_embd)

        residual_cls = RESIDUAL_REGISTRY[config.residual_cls]
        residual_kwargs = config.residual_kwargs

        self.blocks = nn.ModuleList(
            [
                GPT2TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    intermediate_size=config.intermediate_size,
                    dropout_attn=config.dropout_attn,
                    dropout_ffn=config.dropout_ffn,
                    qk_norm=config.qk_norm,
                    attn_bias=config.attn_bias,
                    mlp_bias=config.mlp_bias,
                    mlp_activation=config.mlp_activation,
                    mlp_gated=config.mlp_gated,
                    attn_implementation=config.attn_implementation,
                    layer_idx=i,
                    residual_cls=residual_cls,
                    residual_kwargs=residual_kwargs,
                )
                for i in range(config.n_layers)
            ]
        )

        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model, self.padded_vocab_size, bias=config.lm_head_bias
        )

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask=None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        x = self.dropout_emb(self.pos_emb(self.token_emb(idx)))

        ctx = []
        for block in self.blocks:
            x, ctx = block(x, ctx, attn_mask=attn_mask)

        x = self.ln_f(x)
        if return_logits:
            return self.lm_head(x), None
        return x
