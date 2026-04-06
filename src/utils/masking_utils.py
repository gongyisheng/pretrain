import torch


def build_position_ids(x: torch.Tensor, eot_token_id: int, packing: bool = True) -> torch.Tensor:
    """Compute per-token position IDs from a token sequence.

    packing=True (default): intra-document positions that reset to 0 at the
    token immediately following each EOT. All values are >= 0.

    packing=False: sequential positions 0, 1, 2, ... for the real document
    tokens, and -1 for padding tokens (tokens after the first EOT in the
    sequence). Negative values signal padding so callers can derive a loss mask
    via ``position_ids >= 0``.

    Args:
        x: token IDs, shape (B, S)
        eot_token_id: token ID of the end-of-text / padding token
        packing: True for packed multi-doc sequences, False for single-doc
            padded sequences

    Returns:
        position_ids: shape (B, S), dtype long
    """
    if packing:
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

    # packing=False: sequential positions with -1 for padding.
    # "Padding" = every token strictly after the first EOT in the sequence.
    # The first EOT itself is the document-ending token and counts as real content.
    B, S = x.shape
    is_eot = (x == eot_token_id)
    # seen_eot[b, i] = True iff an EOT appears at some position j < i
    seen_eot = torch.zeros_like(x, dtype=torch.bool)
    seen_eot[:, 1:] = is_eot[:, :-1].cumsum(dim=1).clamp(max=1).bool()
    pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
    return torch.where(seen_eot, torch.full_like(pos, -1), pos)


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
