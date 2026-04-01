import torch


def build_position_ids(x: torch.Tensor, eot_token_id: int) -> torch.Tensor:
    """Compute per-token intra-document position IDs from a packed token sequence.

    position_ids[b, i] = position of token i within its document (resets to 0
    at the token immediately following each EOT token).

    Args:
        x: token IDs, shape (B, S)
        eot_token_id: token ID of the end-of-text token

    Returns:
        position_ids: shape (B, S), dtype long
    """
    is_eot = (x == eot_token_id)
    is_doc_start = torch.zeros_like(x, dtype=torch.bool)
    is_doc_start[:, 1:] = is_eot[:, :-1]
    doc_start_pos = torch.where(
        is_doc_start,
        torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand_as(x),
        torch.zeros_like(x),
    )
    doc_start_cummax, _ = torch.cummax(doc_start_pos, dim=1)
    return torch.arange(x.shape[1], device=x.device).unsqueeze(0) - doc_start_cummax


def build_causal_mask(
    position_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a block-causal additive attention mask from position_ids.

    Exploits the invariant that ``position_ids[b, i] - i`` is constant within
    each document (equals the negative sequence-level start index of that doc).
    Two tokens attend iff they share that constant AND j <= i (causal).

    **Precondition:** ``position_ids`` must contain intra-document positions
    starting from 0 at each document boundary — i.e. the values produced by
    ``build_position_ids``. Passing absolute sequence positions or
    any other encoding will silently produce an incorrect mask.

    Args:
        position_ids: shape (B, S), dtype long — per-token intra-doc position
            (resets to 0 at the token immediately following each EOT token)
        device: target device
        dtype: dtype matching query tensors (required by SDPA)

    Returns:
        additive mask of shape (B, 1, S, S) — 0.0 for attended positions, -inf otherwise
    """
    B, S = position_ids.shape
    adj = position_ids - torch.arange(S, device=device)       # (B, S), constant per doc
    same_doc = (adj.unsqueeze(2) == adj.unsqueeze(1))         # (B, S, S)
    causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()
    attend = same_doc & causal
    additive = torch.zeros(B, 1, S, S, dtype=dtype, device=device)
    additive.masked_fill_(~attend.unsqueeze(1), float('-inf'))
    return additive
