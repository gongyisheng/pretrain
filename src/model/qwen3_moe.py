import torch
import torch.nn as nn

from src.layers.attention import GroupedQueryAttention
from src.layers.block import BaseTransformerBlock
from src.layers.moe import SparseMoEBlock
from src.layers.norm import RMSNorm
from src.layers.pos_emb import RoPE
from src.layers.residual import RESIDUAL_REGISTRY
from src.utils.config import ModelConfig



class Qwen3MoETransformerBlock(BaseTransformerBlock):
    """Qwen3 MoE block: GQA attention + SparseMoE FFN, both wrapped by the
    block's residual strategy. Forward returns (x, ctx, aux_loss) — the
    extra aux_loss is produced by the MoE FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        intermediate_size: int,
        n_experts: int,
        n_experts_per_token: int,
        dropout_attn: float,
        dropout_ffn: float,
        qk_norm: bool = False,
        capacity_factor: float = None,
        attn_bias: bool = False,
        mlp_bias: bool = False,
        **kwargs,
    ):
        super().__init__(d_model, **kwargs)
        self.ln1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_attn, qk_norm, bias=attn_bias)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SparseMoEBlock(d_model, intermediate_size, n_experts, n_experts_per_token, dropout_ffn, capacity_factor, bias=mlp_bias)

    def attn_sublayer(
        self,
        x: torch.Tensor,
        rope: RoPE = None,
        position_ids: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.attn(self.ln1(x), rope, position_ids=position_ids, attn_mask=attn_mask)

    def forward(
        self,
        x: torch.Tensor,
        ctx=None,
        **kwargs,
    ) -> tuple:
        # Attn slot: pre → attn → combine.
        h = self.attn_res_layer.pre(x, ctx)
        attn_out = self.attn_sublayer(h, **kwargs)
        x, ctx = self.attn_res_layer(x, attn_out, ctx)
        # MLP slot: pre → moe ffn (returns aux) → combine.
        h = self.mlp_res_layer.pre(x, ctx)
        ffn_out, aux_loss = self.ffn(self.ln2(h))
        x, ctx = self.mlp_res_layer(x, ffn_out, ctx)
        return x, ctx, aux_loss


class Qwen3MoEModel(nn.Module):
    is_moe = True

    def __init__(self, config: ModelConfig, max_seq_len: int = 2048):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout_emb = nn.Dropout(config.dropout_embd)
        self.rope = RoPE(config.d_model // config.n_heads, max_seq_len, config.rope_theta)

        residual_cls = RESIDUAL_REGISTRY[config.residual_cls]
        residual_kwargs = config.residual_kwargs

        self.blocks = nn.ModuleList([
            Qwen3MoETransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                intermediate_size=config.moe_intermediate_size,
                n_experts=config.moe_n_experts,
                n_experts_per_token=config.moe_n_experts_per_token,
                dropout_attn=config.dropout_attn,
                dropout_ffn=config.dropout_ffn,
                qk_norm=config.qk_norm,
                capacity_factor=config.moe_expert_capacity_factor,
                attn_bias=config.attn_bias,
                mlp_bias=config.mlp_bias,
                layer_idx=i,
                residual_cls=residual_cls,
                residual_kwargs=residual_kwargs,
            )
            for i in range(config.n_layers)
        ])

        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.lm_head_bias)
        if config.tie_word_embeddings:
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
            w1 = module.expert_gate_up if module.gated else module.expert_up
            torch.nn.init.normal_(w1, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.expert_down, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, position_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple:
        """Returns (logits, aux_loss).

        aux_loss is the raw accumulated load-balancing loss across all MoE blocks.
        The caller (trainer) scales it by config.moe_aux_loss_coef before adding
        to the cross-entropy loss.
        """
        x = self.dropout_emb(self.token_emb(idx))
        aux_loss = torch.tensor(0.0, device=idx.device)
        ctx = []
        for block in self.blocks:
            x, ctx, block_aux = block(
                x, ctx, rope=self.rope, position_ids=position_ids, attn_mask=attn_mask,
            )
            aux_loss = aux_loss + block_aux
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, aux_loss
