import math
from functools import partial

import torch.nn as nn

from src.layers.block import TransformerBlock
from src.layers.mlp import MoERouter, SparseMoEBlock
from src.layers.norm import NORM_REGISTRY
from src.layers.pos_emb import POS_EMB_REGISTRY
from src.utils.config import ModelConfig


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig, max_seq_len: int = 1024):
        super().__init__()
        self.config = config

        # Pad vocab to multiple of 128 for better matmul alignment.
        pad = 128
        self.padded_vocab_size = ((config.vocab_size + pad - 1) // pad) * pad
        self.token_emb = nn.Embedding(self.padded_vocab_size, config.d_model)
        self.dropout_emb = nn.Dropout(config.dropout_embd)

        pos_cls = POS_EMB_REGISTRY[config.pos_emb_cls]
        if pos_cls.rotary:
            # Rope head-dim agreement across layers is validated in ModelConfig,
            # so layer 0's is the shared value: qk_rope_head_dim for mla (only
            # its decoupled rope part rotates), else d_model // n_heads.
            attn_cls, attn_kwargs = config.resolve_attn(0)
            head_dim = (
                attn_kwargs["qk_rope_head_dim"]
                if attn_cls == "mla"
                else config.d_model // attn_kwargs["n_heads"]
            )
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
        scaled_weights = set()
        for block in self.blocks:
            for component in (block.attn, block.mlp):
                for module in component.modules():
                    if isinstance(module, nn.Linear):
                        scaled_weights.add(module.weight)
                    elif isinstance(module, SparseMoEBlock):
                        scaled_weights.update((module.expert_up, module.expert_down))
                        if module.gated:
                            scaled_weights.add(module.expert_gate)
                for module in component.modules():
                    if isinstance(module, MoERouter):
                        scaled_weights.discard(module.gate.weight)

        self.apply(partial(self._init_weights, scaled_weights=scaled_weights))
        if config.tie_word_embeddings:
            # Retain the legacy final draw for tied embeddings without reinitializing it.
            self.token_emb.weight = self.lm_head.weight

    def _init_weights(self, module, scaled_weights):
        config = self.config
        weights = []
        if isinstance(module, (nn.Linear, nn.Embedding)):
            weights.append(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, SparseMoEBlock):
            weights.append(module.expert_up)
            if module.gated:
                weights.append(module.expert_gate)
            weights.append(module.expert_down)
        for weight in weights:
            std = config.init_std
            if weight in scaled_weights and config.init_method != "fixed":
                fan_out, fan_in = weight.shape[-2:]
                if config.init_method == "lecun":
                    std = 1 / math.sqrt(fan_in)
                elif config.init_method == "xavier":
                    std = math.sqrt(2 / (fan_in + fan_out))
                else:
                    std = math.sqrt(2 / fan_in)
            if config.init_distribution == "normal":
                nn.init.normal_(weight, mean=0.0, std=std)
            else:
                bound = math.sqrt(3) * std
                nn.init.uniform_(weight, -bound, bound)

    @staticmethod
    def compute_flops(config: ModelConfig, max_seq_len: int) -> int:
        """Forward FLOPs per token: the sum of each block's `compute_flops` plus
        the final norm and lm_head. Summed per-layer (not n_layers × one block)
        because some residual strategies are depth-dependent. The backward
        multiplier is applied by the caller. Embedding lookup and RoPE are 0.
        """
        blocks = sum(
            TransformerBlock.compute_flops(config, max_seq_len, i)
            for i in range(config.n_layers)
        )
        final_norm = 3 * config.d_model
        lm_head = 2 * config.d_model * config.vocab_size + (
            config.vocab_size if config.lm_head_bias else 0
        )
        return blocks + final_norm + lm_head

    @staticmethod
    def compute_parameters(config: ModelConfig, max_seq_len: int) -> dict[str, int]:
        """Analytic {total, non_emb, active_non_emb} param counts, mirroring
        `compute_parameters` summed up from each block. Reproduces the live model's
        `sum(p.numel())` exactly: vocab is padded to a multiple of 128, tied
        lm_head shares the token-embedding weight, and learned pos_emb is counted
        as an embedding (excluded from non_emb). For MoE, active_non_emb counts
        only the k routed experts per layer; for dense it equals non_emb.
        """
        d_model = config.d_model
        pad = 128
        padded_vocab = ((config.vocab_size + pad - 1) // pad) * pad

        pos = POS_EMB_REGISTRY[config.pos_emb_cls].compute_parameters(
            max_seq_len, d_model
        )
        embeddings = padded_vocab * d_model + pos  # token_emb + (learned) pos_emb

        # Blocks may differ per layer (mixed dense/MoE stacks), so sum per layer
        # rather than scaling one block's count by n_layers.
        blocks = sum(
            TransformerBlock.compute_parameters(config, layer_idx=i)
            for i in range(config.n_layers)
        )
        blocks_active = sum(
            TransformerBlock.compute_parameters(config, active=True, layer_idx=i)
            for i in range(config.n_layers)
        )
        final_norm = NORM_REGISTRY[config.norm_cls].compute_parameters(
            d_model, **config.norm_kwargs
        )
        # Tied → lm_head.weight is the token_emb weight (already in embeddings).
        lm_head = (0 if config.tie_word_embeddings else padded_vocab * d_model) + (
            padded_vocab if config.lm_head_bias else 0
        )

        non_emb = blocks + final_norm + lm_head
        active_non_emb = blocks_active + final_norm + lm_head
        return {
            "total": embeddings + non_emb,
            "non_emb": non_emb,
            "active_non_emb": active_non_emb,
        }

    def forward(self, idx, position_ids, attn_masks, return_logits=True):
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        x = self.dropout_emb(x)

        aux_total = None
        ctx = []
        for layer_idx, block in enumerate(self.blocks):
            x, ctx, aux_loss = block(
                x,
                ctx,
                rope=self.rope,
                position_ids=position_ids,
                attn_mask=attn_masks[layer_idx],
            )
            if aux_loss is not None:
                aux_total = aux_loss if aux_total is None else aux_total + aux_loss

        x = self.ln_f(x)
        if not return_logits:
            return x
        return self.lm_head(x), aux_total

    def post_step(self):
        for module in self.modules():
            if module is not self and hasattr(module, "post_step"):
                module.post_step()

    def forward_meta(self) -> list[dict]:
        """Per-layer metadata produced by the forward pass (side-channel to the
        returned logits/aux loss), one dict per sublayer that exposes it, in
        layer order. Dispatches to each sublayer's forward_meta like post_step;
        [] for dense models.

        Mode follows self.training: in train mode, routing counts accumulated
        across micro-batches (reset per optim step); in eval mode, the current
        batch's counts.
        """
        meta = []
        for module in self.modules():
            if module is not self and hasattr(module, "forward_meta"):
                meta.append(module.forward_meta())
        return meta
