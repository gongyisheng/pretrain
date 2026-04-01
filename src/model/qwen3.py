import torch
import torch.nn as nn

from src.model.components import (
    BaseTransformerBlock,
    build_doc_causal_mask_from_position_ids,
    GroupedQueryAttention,
    RMSNorm,
    RoPE,
    SwiGluFFN,
)
from src.utils.config import ModelConfig


class Qwen3TransformerBlock(BaseTransformerBlock):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        intermediate_size: int,
        dropout: float,
        qk_norm: bool = False,
        **kwargs,
    ):
        super().__init__(d_model, **kwargs)
        self.ln1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, dropout, qk_norm
        )
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGluFFN(d_model, intermediate_size, dropout)

    def attn_sublayer(
        self,
        x: torch.Tensor,
        rope: RoPE,
        attn_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.attn(
            self.ln1(x), rope, attn_mask=attn_mask, position_ids=position_ids
        )

    def ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ln2(x))


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig, max_seq_len: int = 2048):
        super().__init__()
        self.config = config

        # Pad vocab to multiple of 128 for better matmul alignment
        pad_multiple = 128
        self.padded_vocab_size = (
            (config.vocab_size + pad_multiple - 1) // pad_multiple
        ) * pad_multiple
        self.token_emb = nn.Embedding(self.padded_vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        # No pos_emb — positioning handled by RoPE inside each attention layer
        self.rope = RoPE(
            config.d_model // config.n_heads, max_seq_len, config.rope_theta
        )

        self.blocks = nn.ModuleList(
            [
                Qwen3TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    qk_norm=config.qk_norm,
                    attn_res=config.attn_res,
                    attn_res_block_size=config.attn_res_block_size,
                    attn_res_norm=config.attn_res_norm,
                    layer_idx=i,
                )
                for i in range(config.n_layers)
            ]
        )

        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, self.padded_vocab_size, bias=False)

        # Weight tying
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
        position_ids: torch.Tensor = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        x = self.drop(self.token_emb(idx))

        attn_mask = (
            build_doc_causal_mask_from_position_ids(position_ids, idx.device, x.dtype)
            if position_ids is not None
            else None
        )

        if self.config.attn_res:
            attn_res_ctx = []
            for block in self.blocks:
                x, attn_res_ctx = block(
                    x,
                    attn_res_ctx,
                    rope=self.rope,
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                )
        else:
            for block in self.blocks:
                x = block(
                    x, rope=self.rope, attn_mask=attn_mask, position_ids=position_ids
                )

        x = self.ln_f(x)
        if return_logits:
            return self.lm_head(x)
        return x
