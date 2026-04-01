import torch


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
    ``Trainer._build_position_ids``. Passing absolute sequence positions or
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
