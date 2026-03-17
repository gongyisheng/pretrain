import torch
import torch.nn as nn

from src.model.components import GroupedQueryAttention, RMSNorm, SwiGluFFN
from src.utils.config import ModelConfig


class Qwen3TransformerBlock(nn.Module):
    """Pre-RMSNorm transformer block with GQA, SwiGLU FFN, and RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        rope_theta: float,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout, max_seq_len, rope_theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGluFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig, max_seq_len: int = 2048):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        # No pos_emb — positioning handled by RoPE inside each attention layer

        self.blocks = nn.ModuleList([
            Qwen3TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                max_seq_len=max_seq_len,
                rope_theta=config.rope_theta,
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.token_emb(idx))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)
