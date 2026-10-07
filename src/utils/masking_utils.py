import torch
from torch.nn.attention.flex_attention import create_block_mask


def build_position_ids(
    x: torch.Tensor, eot_token_id: int, packing: bool = True
) -> torch.Tensor:
    """Build position IDs for packed or padded token sequences."""
    if packing:
        is_eot = x == eot_token_id
        is_doc_start = torch.zeros_like(x, dtype=torch.bool)
        is_doc_start[:, 1:] = is_eot[:, :-1]
        doc_start_pos = torch.where(
            is_doc_start,
            torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand_as(x),
            torch.zeros_like(x),
        )
        doc_start_cummax, _ = torch.cummax(doc_start_pos, dim=1)
        return torch.arange(x.shape[1], device=x.device).unsqueeze(0) - doc_start_cummax

    # Tokens after the first EOT are padding (-1).
    B, S = x.shape
    is_eot = x == eot_token_id
    seen_eot = torch.zeros_like(x, dtype=torch.bool)
    seen_eot[:, 1:] = is_eot[:, :-1].cumsum(dim=1).clamp(max=1).bool()
    pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
    return torch.where(seen_eot, torch.full_like(pos, -1), pos)


def build_intra_doc_attention_mask(
    position_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
):
    """Build a causal attention mask restricted to each document."""
    if attn_implementation == "flex_attention":
        return _build_attention_mask_for_flex_attn(position_ids, device)
    return _build_attention_mask_for_sdpa(position_ids, device, dtype)


def build_causal_attention_mask(
    B: int,
    S: int,
    device: torch.device,
    attn_implementation: str,
):
    """Build a causal attention mask without document separation."""
    if attn_implementation == "sdpa":
        return None
    # Sequential positions treat each sequence as one document.
    causal_pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    return _build_attention_mask_for_flex_attn(causal_pos, device)


def _build_attention_mask_for_sdpa(
    position_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    B, S = position_ids.shape
    # This offset is constant within each document.
    adj = position_ids - torch.arange(S, device=device)
    same_doc = adj.unsqueeze(2) == adj.unsqueeze(1)
    causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()
    attend = same_doc & causal
    additive = torch.zeros(B, 1, S, S, dtype=dtype, device=device)
    additive.masked_fill_(~attend.unsqueeze(1), float("-inf"))
    return additive


_create_block_mask_compiled = torch.compile(create_block_mask)


def _build_attention_mask_for_flex_attn(
    position_ids: torch.Tensor, device: torch.device
):
    B, S = position_ids.shape
    # This offset is constant within each document.
    adj = torch.arange(S, device=device).unsqueeze(0) - position_ids

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (adj[b, q_idx] == adj[b, kv_idx])

    return _create_block_mask_compiled(
        mask_mod,
        B=B,
        H=None,
        Q_LEN=S,
        KV_LEN=S,
        device=device,
    )
