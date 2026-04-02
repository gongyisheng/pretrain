import torch
import torch.nn as nn

from src.model.components import GroupedQueryAttention, RMSNorm, RoPE, SparseMoEBlock
from src.utils.config import ModelConfig


class Qwen3MoETransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        moe_intermediate_size: int,
        n_experts: int,
        n_experts_per_token: int,
        dropout_attn: float,
        dropout_ffn: float,
        qk_norm: bool = False,
        capacity_factor: float = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_attn, qk_norm)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SparseMoEBlock(d_model, moe_intermediate_size, n_experts, n_experts_per_token, dropout_ffn, capacity_factor)

    def forward(self, x: torch.Tensor, rope: RoPE) -> tuple:
        x = x + self.attn(self.ln1(x), rope)
        ffn_out, aux_loss = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x, aux_loss


class Qwen3MoEModel(nn.Module):
    is_moe = True

    def __init__(self, config: ModelConfig, max_seq_len: int = 2048):
        super().__init__()
        if config.attn_res:
            raise ValueError("Qwen3MoEModel does not support attn_res. Set attn_res=False.")
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout_embedding)
        self.rope = RoPE(config.d_model // config.n_heads, max_seq_len, config.rope_theta)

        self.blocks = nn.ModuleList([
            Qwen3MoETransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                moe_intermediate_size=config.moe_intermediate_size,
                n_experts=config.n_experts,
                n_experts_per_token=config.n_experts_per_token,
                dropout_attn=config.dropout_attn,
                dropout_ffn=config.dropout_ffn,
                qk_norm=config.qk_norm,
                capacity_factor=config.moe_expert_capacity_factor,
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, SparseMoEBlock):
            torch.nn.init.normal_(module.expert_gate_up, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.expert_down, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple:
        """Returns (logits, aux_loss).

        aux_loss is the raw accumulated load-balancing loss across all MoE blocks.
        The caller (trainer) scales it by config.moe_aux_loss_coef before adding
        to the cross-entropy loss.
        """
        x = self.drop(self.token_emb(idx))
        aux_loss = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x, block_aux = block(x, rope=self.rope)
            aux_loss = aux_loss + block_aux
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, aux_loss
