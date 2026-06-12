import torch.nn as nn

from src.layers.attention import ATTN_REGISTRY
from src.layers.block import BaseTransformerBlock
from src.layers.mlp import MLP_REGISTRY, SparseMoEBlock
from src.layers.norm import NORM_REGISTRY
from src.layers.pos_emb import POS_EMB_REGISTRY
from src.layers.residual import RESIDUAL_REGISTRY
from src.utils.config import ModelConfig


class TransformerBlock(BaseTransformerBlock):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(
            config.d_model,
            layer_idx,
            RESIDUAL_REGISTRY[config.residual_cls],
            config.residual_kwargs,
        )
        norm_cls = NORM_REGISTRY[config.norm_cls]
        attn_cls = ATTN_REGISTRY[config.attn_cls]
        mlp_cls = MLP_REGISTRY[config.mlp_cls]
        self.norm1 = norm_cls(config.d_model, **config.norm_kwargs)
        self.attn = attn_cls(config.d_model, **config.attn_kwargs)
        self.norm2 = norm_cls(config.d_model, **config.norm_kwargs)
        self.mlp = mlp_cls(config.d_model, **config.mlp_kwargs)

    def attn_sublayer(self, x, rope=None, position_ids=None, attn_mask=None):
        return self.attn(
            self.norm1(x), rope=rope, position_ids=position_ids, attn_mask=attn_mask
        )

    def mlp_sublayer(self, x):
        return self.mlp(self.norm2(x))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig, max_seq_len: int = 1024):
        super().__init__()
        self.config = config
        self.is_moe = config.mlp_cls == "moe"

        # Pad vocab to multiple of 128 for better matmul alignment.
        pad = 128
        self.padded_vocab_size = ((config.vocab_size + pad - 1) // pad) * pad
        self.token_emb = nn.Embedding(self.padded_vocab_size, config.d_model)
        self.dropout_emb = nn.Dropout(config.dropout_embd)

        pos_cls = POS_EMB_REGISTRY[config.pos_emb_cls]
        if pos_cls.rotary:
            head_dim = config.d_model // config.attn_kwargs["n_heads"]
            self.rope = pos_cls(head_dim, max_seq_len, **config.pos_emb_kwargs)
            self.pos_emb = None
        else:
            self.pos_emb = pos_cls(max_seq_len, config.d_model, **config.pos_emb_kwargs)
            self.rope = None

        self.blocks = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.n_layers)]
        )
        norm_cls = NORM_REGISTRY[config.norm_cls]
        self.ln_f = norm_cls(config.d_model, **config.norm_kwargs)
        self.lm_head = nn.Linear(
            config.d_model, self.padded_vocab_size, bias=config.lm_head_bias
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, SparseMoEBlock):
            w1 = module.expert_gate_up if module.gated else module.expert_up
            nn.init.normal_(w1, mean=0.0, std=0.02)
            nn.init.normal_(module.expert_down, mean=0.0, std=0.02)

    def forward(self, idx, position_ids, attn_mask=None, return_logits=True):
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        x = self.dropout_emb(x)

        aux_total = None
        ctx = []
        for block in self.blocks:
            x, ctx, aux = block(
                x, ctx, rope=self.rope, position_ids=position_ids, attn_mask=attn_mask
            )
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux

        x = self.ln_f(x)
        if not return_logits:
            return x
        return self.lm_head(x), aux_total
