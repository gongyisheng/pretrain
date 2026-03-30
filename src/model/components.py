import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Backend selection ---
# "torch" uses @torch.compile fused kernels; "triton" uses hand-written Triton kernels.
# Kernel implementations live in src/kernel/{torch,triton}/.

_rmsnorm    = None
_rope       = None
_swiglu     = None
_flash_attn = None


def set_backend(backend: str):
    global _rmsnorm, _rope, _swiglu, _flash_attn
    if backend == "torch":
        from src.kernel.torch.rmsnorm import torch_rmsnorm
        from src.kernel.torch.rope import torch_rope
        from src.kernel.torch.swiglu import torch_swiglu
        from src.kernel.torch.flashattn import torch_flash_attn
        _rmsnorm, _rope, _swiglu, _flash_attn = torch_rmsnorm, torch_rope, torch_swiglu, torch_flash_attn
    elif backend == "triton":
        from src.kernel.torch.rmsnorm import torch_rmsnorm
        from src.kernel.triton.rope import triton_rope
        from src.kernel.triton.swiglu import triton_swiglu
        from src.kernel.triton.flashattn import triton_flash_attn
        _rmsnorm, _rope, _swiglu, _flash_attn = torch_rmsnorm, triton_rope, triton_swiglu, triton_flash_attn
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'torch' or 'triton'.")


@torch.compile(dynamic=True)
def _attn_res_aggregate_rmsnorm(
    V: torch.Tensor, w_proj: torch.Tensor, norm_weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fused: RMSNorm keys → dot-product logits → softmax → weighted sum.

    Inlining the norm eliminates the K tensor write/read to HBM.
    dynamic=True because N grows across layers.
    """
    dtype = V.dtype
    vf = V.float()
    K = (vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)).to(dtype) * norm_weight
    logits  = (K * w_proj).sum(-1)         # (N, B, S)
    weights = logits.softmax(0)            # (N, B, S)
    return (weights.unsqueeze(-1) * V).sum(0)  # (B, S, D)


@torch.compile(dynamic=True)
def _attn_res_aggregate_layernorm(
    V: torch.Tensor, w_proj: torch.Tensor,
    norm_weight: torch.Tensor, norm_bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """Fused: LayerNorm keys → dot-product logits → softmax → weighted sum."""
    dtype = V.dtype
    vf   = V.float()
    mean = vf.mean(-1, keepdim=True)
    K    = ((vf - mean) * torch.rsqrt((vf - mean).pow(2).mean(-1, keepdim=True) + eps)).to(dtype) * norm_weight + norm_bias
    logits  = (K * w_proj).sum(-1)
    weights = logits.softmax(0)
    return (weights.unsqueeze(-1) * V).sum(0)


# --- Norms ---

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        out = _rmsnorm(x.reshape(-1, orig_shape[-1]), self.weight, self.eps)
        return out.reshape(orig_shape)


# --- RoPE ---

class RoPE(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.theta = theta
        self.max_seq_len = max_seq_len
        self._build_buffers()

    def _build_buffers(self):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_head, 2) / self.d_head))
        positions = torch.arange(self.max_seq_len)
        angles = positions[:, None] * freqs[None, :]          # (max_seq_len, d_head//2)
        angles = torch.cat([angles, angles], dim=-1)           # (max_seq_len, d_head)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, S, d_head)
        S = x.shape[2]
        cos = self.cos[:S][None, None, :, :].to(x.dtype)      # (1, 1, S, d_head)
        sin = self.sin[:S][None, None, :, :].to(x.dtype)      # (1, 1, S, d_head)
        return _rope(x, cos, sin)


# --- Attention ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, S, d_head)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q.reshape(-1, S, self.d_head)).view(B, self.n_heads, S, self.d_head)
            k = self.k_norm(k.reshape(-1, S, self.d_head)).view(B, self.n_heads, S, self.d_head)

        out = _flash_attn(q, k, v, causal=True)

        out = out.transpose(1, 2).reshape(B, S, H)
        return self.resid_dropout(self.out_proj(out))


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor, rope: RoPE) -> torch.Tensor:
        B, S, H = x.shape

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)    # (B, n_heads, S, d_head)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2) # (B, n_kv_heads, S, d_head)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2) # (B, n_kv_heads, S, d_head)

        if self.qk_norm:
            q = self.q_norm(q.reshape(-1, S, self.d_head)).view(B, self.n_heads, S, self.d_head)
            k = self.k_norm(k.reshape(-1, S, self.d_head)).view(B, self.n_kv_heads, S, self.d_head)

        q = rope(q)
        k = rope(k)

        # Expand KV heads for GQA (expand+reshape avoids memory allocation vs repeat_interleave)
        k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_groups, S, self.d_head).reshape(B, self.n_heads, S, self.d_head)
        v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_groups, S, self.d_head).reshape(B, self.n_heads, S, self.d_head)
        out = _flash_attn(q, k, v, causal=True)

        out = out.transpose(1, 2).reshape(B, S, H)
        return self.resid_dropout(self.out_proj(out))


# --- FFN ---

class GeluFFN(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SwiGluFFN(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.down_proj(_swiglu(gate, up))
        x = self.dropout(x)
        return x


# --- MoE ---

class MoERouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, n_experts_per_token: int, normalize: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.normalize = normalize
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (T, d_model)  where T = B*S (flattened)
        logits = self.gate(x)                                        # (T, n_experts)
        router_probs = logits.softmax(-1)                            # (T, n_experts)
        top_weights, top_indices = torch.topk(router_probs, self.n_experts_per_token, dim=-1)
        if self.normalize:
            top_weights = top_weights / (top_weights.sum(-1, keepdim=True) + 1e-9)
        return top_indices, top_weights, router_probs


class SparseMoEBlock(nn.Module):
    """Sparse Mixture-of-Experts FFN block.

    Each expert is a SwiGluFFN. For each forward pass:
      - Tokens are routed to top-k experts via MoERouter.
      - Each expert processes only its assigned tokens.
      - Outputs are weighted and summed back into the full sequence.
      - A load-balancing auxiliary loss (Switch Transformer formula) is returned.

    Note: aux_loss scale grows linearly with n_experts_per_token (k). Under balanced
    routing the expected value is approximately k. Callers using moe_aux_loss_coef should
    account for this when comparing runs across different k values.

    Note: forward() is not torch.compile-compatible due to dynamic expert dispatch
    (boolean index masking produces variable-size tensors that cause graph breaks).
    The trainer must NOT wrap this model with torch.compile.
    """

    def __init__(self, d_model: int, intermediate_size: int, n_experts: int, n_experts_per_token: int, dropout: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.router = MoERouter(d_model, n_experts, n_experts_per_token)
        self.experts = nn.ModuleList([
            SwiGluFFN(d_model, intermediate_size, dropout) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor):
        # x: (B, S, D)
        B, S, D = x.shape
        T = B * S
        x_flat = x.view(T, D)                          # (T, D)

        top_indices, top_weights, router_probs = self.router(x_flat)
        # top_indices: (T, k)   top_weights: (T, k)   router_probs: (T, n_experts)

        output = torch.zeros_like(x_flat)              # (T, D)

        for expert_idx in range(self.n_experts):
            # mask[t, slot] = True when token t is assigned to this expert at that slot
            mask = (top_indices == expert_idx)          # (T, k) bool
            token_mask = mask.any(dim=-1)               # (T,) bool — any slot routes here
            if not token_mask.any():
                continue
            # Per-token weight from this expert: sum over matched slots
            weight = (top_weights * mask.float()).sum(-1)   # (T,)

            expert_out = self.experts[expert_idx](x_flat[token_mask])   # (n_routed, D)
            output[token_mask] += expert_out * weight[token_mask].unsqueeze(-1)

        # Switch Transformer load-balancing auxiliary loss:
        #   L = n_experts * sum_i(f_i * P_i)
        #   f_i = fraction of tokens routed to expert i (hard, no grad)
        #   P_i = mean router probability for expert i (soft, carries grad through router)
        with torch.no_grad():
            # One-hot encode routing decisions -> (T, n_experts)
            expert_mask = torch.zeros(T, self.n_experts, device=x.device, dtype=x.dtype)
            for k in range(self.n_experts_per_token):
                expert_mask.scatter_add_(
                    1,
                    top_indices[:, k].unsqueeze(1),
                    torch.ones(T, 1, device=x.device, dtype=x.dtype),
                )
            f = expert_mask.mean(0)                     # (n_experts,) fraction per expert

        P = router_probs.mean(0)                        # (n_experts,) mean softmax prob per expert
        aux_loss = self.n_experts * (f * P).sum()

        return output.view(B, S, D), aux_loss


# --- Transformer Block ---

def _block_attn_res(
    attn_res_ctx: list,
    x: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    """Compute block-level attention residual.

    Args:
        attn_res_ctx: list of finalized block tensors, each shape (B, S, D)
        x:   current partial block (hidden state), shape (B, S, D)
        proj: Linear(d_model, 1, bias=False) — learned query vector w_l
        norm: RMSNorm or LayerNorm applied to keys before attention

    Returns:
        Attention-weighted combination of all blocks + x, shape (B, S, D)
    """
    V = torch.stack(attn_res_ctx + [x])            # (N+1, B, S, D)
    w = proj.weight.view(-1)
    if isinstance(norm, nn.LayerNorm):
        return _attn_res_aggregate_layernorm(V, w, norm.weight, norm.bias, norm.eps)
    return _attn_res_aggregate_rmsnorm(V, w, norm.weight, norm.eps)

class BaseTransformerBlock(nn.Module):
    """Base transformer block with optional AttnRes support.

    Subclasses implement attn_sublayer() and ffn_sublayer().
    The residual logic (standard or AttnRes) lives here once for all architectures.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = 0,
        attn_res: bool = False,
        attn_res_block_size: int = 1,
        attn_res_norm: str = "rmsnorm",
    ):
        super().__init__()
        self.attn_res = attn_res
        if attn_res:
            self.layer_idx = layer_idx
            self.attn_res_block_size = attn_res_block_size
            norm_cls = nn.LayerNorm if attn_res_norm == "layernorm" else RMSNorm
            self.attn_res_proj = nn.Linear(d_model, 1, bias=False)
            self.attn_res_norm = norm_cls(d_model)
            self.mlp_res_proj = nn.Linear(d_model, 1, bias=False)
            self.mlp_res_norm = norm_cls(d_model)

    def attn_sublayer(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, attn_res_ctx=None, **kwargs) -> tuple:
        if self.attn_res:
            partial_block = x

            # compute h before attn using current partial_block as current-block rep
            h = _block_attn_res(attn_res_ctx, partial_block, self.attn_res_proj, self.attn_res_norm)

            # seal at block boundary before processing attn; seeds blocks with embedding at layer 0
            if self.layer_idx % self.attn_res_block_size == 0:
                attn_res_ctx = attn_res_ctx + [partial_block]
                partial_block = None

            attn_out = self.attn_sublayer(h, **kwargs)
            partial_block = partial_block + attn_out if partial_block is not None else attn_out

            # compute h before FFN
            h = _block_attn_res(attn_res_ctx, partial_block, self.mlp_res_proj, self.mlp_res_norm)

            mlp_out = self.ffn_sublayer(h)
            partial_block = partial_block + mlp_out

            return partial_block, attn_res_ctx
        else:
            # Standard residual path
            x = x + self.attn_sublayer(x, **kwargs)
            x = x + self.ffn_sublayer(x)
            return x

