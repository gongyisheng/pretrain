import torch
import torch.nn as nn

from src.model.components import BaseTransformerBlock, GeluFFN, MultiHeadAttention
from src.utils.config import ModelConfig
from src.utils.masking_utils import build_causal_mask


class GPT2TransformerBlock(BaseTransformerBlock):
    def __init__(self, d_model: int, n_heads: int, intermediate_size: int, dropout: float, qk_norm: bool = False, **kwargs):
        super().__init__(d_model, **kwargs)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, qk_norm=qk_norm)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = GeluFFN(d_model, intermediate_size, dropout)

    def attn_sublayer(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        return self.attn(self.ln1(x), attn_mask=attn_mask)

    def ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ln2(x))


class GPT2Model(nn.Module):
    def __init__(self, config: ModelConfig, max_seq_len: int = 1024):
        super().__init__()
        self.config = config

        # Pad vocab to multiple of 128 for better matmul alignment
        pad_multiple = 128
        self.padded_vocab_size = ((config.vocab_size + pad_multiple - 1) // pad_multiple) * pad_multiple
        self.token_emb = nn.Embedding(self.padded_vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            GPT2TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                intermediate_size=config.intermediate_size,
                dropout=config.dropout,
                qk_norm=config.qk_norm,
                attn_res=config.attn_res,
                attn_res_block_size=config.attn_res_block_size,
                attn_res_norm=config.attn_res_norm,
                layer_idx=i,
            )
            for i in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
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

    def forward(self, idx: torch.Tensor, position_ids: torch.Tensor = None, return_logits: bool = True) -> torch.Tensor:
        B, S = idx.shape
        # Absolute position embedding always uses 0..S-1, never intra-doc position_ids
        pos = torch.arange(0, S, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))

        if position_ids is None:
            position_ids = torch.arange(S, device=idx.device).unsqueeze(0).expand(B, S)
        attn_mask = build_causal_mask(position_ids, idx.device, x.dtype)

        if self.config.attn_res:
            attn_res_ctx = []
            for block in self.blocks:
                x, attn_res_ctx = block(x, attn_res_ctx, attn_mask=attn_mask)
        else:
            for block in self.blocks:
                x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        if return_logits:
            return self.lm_head(x)
        return x
