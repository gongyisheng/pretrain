import torch

# Flipping this re-specializes the compiled graph. Independent of the activation
# flag because this fold is far costlier, so their cadences are free to diverge.
_RECORDING = [False]


class QuantizationStats(torch.nn.Module):
    """Running quantization-error sums for one GEMM operand over a window.

    `n_groups` is 1 for a dense operand and the expert count for a grouped one,
    where metrics are reduced per expert so a cold expert is not averaged away.
    """

    FIELDS = ("src_sq", "err_sq", "under", "numel", "nonzero")

    def __init__(self, key: str, n_groups: int, device: torch.device):
        super().__init__()
        # Resolved at install time: which GEMM/operand this belongs to is not a
        # module path, so nothing can re-derive it.
        self.key = key
        self.grouped = n_groups > 1
        for name in self.FIELDS:
            self.register_buffer(
                name,
                torch.zeros(n_groups, dtype=torch.float32, device=device),
                persistent=False,
            )

    def reset(self) -> None:
        for name in self.FIELDS:
            getattr(self, name).zero_()


def accumulate_quantization_sums(
    source_tensor: torch.Tensor,
    codes: torch.Tensor,
    dequantized_tensor: torch.Tensor,
    offs: torch.Tensor | None = None,
    ragged_dim: int | None = None,
    contract_dim: int | None = None,
    rotated_source: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return five FP32 sum tensors, grouped by stacks or ragged offsets."""
    source = source_tensor.float()
    mask_source = source if rotated_source is None else rotated_source.float()
    dequantized = dequantized_tensor.float()
    if codes.dtype is torch.uint8:
        if contract_dim not in (-2, -1):
            raise ValueError("packed fp4 codes require contract_dim -2 or -1")
        moved = codes.movedim(contract_dim, -1)
        zero_codes = (
            torch.stack(((moved & 0x7) == 0, (moved & 0x70) == 0), dim=-1)
            .flatten(-2)
            .movedim(-1, contract_dim)
        )
    else:
        zero_codes = codes.float() == 0
    squares = source.square()
    err_squares = (source - dequantized).square()
    nonzero_mask = mask_source != 0
    underflows = (nonzero_mask & zero_codes).float()

    if offs is None:
        src_sq = squares.sum().reshape(1)
        err_sq = err_squares.sum().reshape(1)
        under = underflows.sum().reshape(1)
        nonzero = nonzero_mask.sum(dtype=torch.float32).reshape(1)
        numel = torch.full_like(src_sq, source.numel())
    elif source.ndim == 3:  # stacked expert weights, expert on dim 0
        src_sq = squares.flatten(1).sum(1)
        err_sq = err_squares.flatten(1).sum(1)
        under = underflows.flatten(1).sum(1)
        nonzero = nonzero_mask.flatten(1).sum(1, dtype=torch.float32)
        numel = torch.full_like(src_sq, source.shape[-2] * source.shape[-1])
    elif source.ndim == 2:
        ragged_dim = -2 if ragged_dim is None else ragged_dim
        if ragged_dim not in (-2, -1):
            raise ValueError(f"ragged_dim must be -2 or -1, got {ragged_dim}")
        n_groups = offs.shape[0]
        ragged_length = source.shape[ragged_dim]
        dense_dim = -1 if ragged_dim == -2 else -2
        positions = torch.arange(ragged_length, device=offs.device)
        # right=True maps a position to the group whose end offset first exceeds it.
        ids = torch.searchsorted(offs, positions, right=True).clamp_(max=n_groups - 1)

        def by_expert(t: torch.Tensor) -> torch.Tensor:
            per_position = t.sum(dense_dim)
            return per_position.new_zeros(n_groups).index_add_(0, ids, per_position)

        src_sq, err_sq, under, nonzero = (
            by_expert(squares),
            by_expert(err_squares),
            by_expert(underflows),
            by_expert(nonzero_mask.float()),
        )
        starts = torch.cat([offs.new_zeros(1), offs[:-1]])
        numel = ((offs - starts) * source.shape[dense_dim]).to(src_sq.dtype)
    else:
        raise ValueError(f"grouped source must be 2D or 3D, got {source.ndim}D")

    return src_sq, err_sq, under, numel, nonzero


def record_operand(
    stats: QuantizationStats | None,
    quantization_stats: torch.Tensor,
) -> None:
    """Fold one returned quantization-statistics tensor into an armed site."""
    if stats is None or not _RECORDING[0]:
        return
    for name, value in zip(stats.FIELDS, quantization_stats.detach().unbind(-1)):
        getattr(stats, name).add_(value)


def set_quantization_monitoring_status(enabled: bool) -> None:
    """Arm or disarm accumulation; a window must span a whole optimizer step."""
    _RECORDING[0] = enabled


def get_quantization_monitoring_status() -> bool:
    """Return whether quantization-statistics accumulation is armed."""
    return _RECORDING[0]


def reset_quantization_stats(model) -> None:
    """Zero every accumulator."""
    for module in getattr(model, "_orig_mod", model).modules():
        if isinstance(module, QuantizationStats):
            module.reset()
