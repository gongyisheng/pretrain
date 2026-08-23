import torch

from src.quant.quantize import dequantize_operand
from src.quant.rotation import Rotation


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
    source_tensor,
    codes,
    dequantized_tensor,
    offs=None,
    rotated_source=None,
):
    """One operand's quantization-error partial sums, for folding into a window.

    Returns `(src_sq, err_sq, under, numel, nonzero)`, each a 1D fp32 tensor:
    length 1 for a dense operand, length n_experts for a grouped one. These are
    sums, not metrics, because sqnr (a log of a ratio) does not accumulate --
    `compute_quantization_metrics` divides once at read time over the window.

    `numel` counts every element (the validity signal for whether an expert got
    tokens); `nonzero` counts elements that could underflow (the rate's denominator).
    `codes == 0` is exact for rounding-to-zero since scales are positive.
    `rotated_source` is the source in the codes' (rotated) basis, used only for the
    underflow/nonzero mask; `source` and `dequantized` stay in the original basis so
    energy and reconstruction error are rotation-agnostic.

    `offs` (cumulative end-offsets) switches the reduction to per-expert so a cold,
    badly scaled expert is not averaged away. Stacked expert weights carry the
    expert on dim 0; dispatched rows are ragged along `offs`.
    """
    source = source_tensor.float()
    mask_source = source if rotated_source is None else rotated_source.float()
    dequantized = dequantized_tensor.float()
    code_values = codes.float()
    squares = source.square()
    err_squares = (source - dequantized).square()
    nonzero_mask = mask_source != 0
    underflows = (nonzero_mask & (code_values == 0)).float()

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
        numel = torch.full_like(src_sq, source[0].numel())
    else:  # dispatched rows, ragged along offs
        n_groups = offs.shape[0]
        rows = torch.arange(source.shape[0], device=offs.device)
        # right=True: row r belongs to the group whose end-offset first exceeds r
        ids = torch.searchsorted(offs, rows, right=True).clamp_(max=n_groups - 1)

        def by_expert(t):
            per_row = t.flatten(1).sum(1)
            return per_row.new_zeros(n_groups).index_add_(0, ids, per_row)

        src_sq, err_sq, under, nonzero = (
            by_expert(squares),
            by_expert(err_squares),
            by_expert(underflows),
            by_expert(nonzero_mask.float()),
        )
        starts = torch.cat([offs.new_zeros(1), offs[:-1]])
        numel = ((offs - starts) * source.shape[1]).to(src_sq.dtype)

    return src_sq, err_sq, under, numel, nonzero


def record_operand(
    stats,
    source,
    codes,
    scale,
    contract_dim,
    scale_cfg,
    offs=None,
    ragged_dim=None,
    rotation: Rotation | None = None,
) -> None:
    """Fold one quantized operand into `stats`, if armed.

    Called right after the quantize that produced `codes`, so `source`/`codes` are
    the exact tensors the GEMM consumes.

    `offs` is passed to the sums unconditionally (per-expert reduction needs the real
    one) but to `dequantize_operand` only on the ragged path (`ragged_dim` set);
    passing it unpaired there is dead weight and trips its given-together check.
    """
    if stats is None or not _RECORDING[0]:
        return
    # Detach first: `codes`/`scale` are live graph nodes, so folding them in would
    # make the accumulator buffers require grad and retain the step's graph.
    source, codes, scale = source.detach(), codes.detach(), scale.detach()
    rotated_source = None if rotation is None else rotation(source, contract_dim)
    dequantized = dequantize_operand(
        codes,
        scale,
        contract_dim,
        scale_cfg,
        offs=offs if ragged_dim is not None else None,
        ragged_dim=ragged_dim,
        rotation=rotation,
    )
    sums = accumulate_quantization_sums(
        source,
        codes,
        dequantized,
        offs=offs,
        rotated_source=rotated_source,
    )
    for name, value in zip(stats.FIELDS, sums):
        getattr(stats, name).add_(value)


def set_quantization_monitoring_status(enabled: bool) -> None:
    """Arm or disarm accumulation; a window must span a whole optimizer step."""
    _RECORDING[0] = enabled


def reset_quantization_stats(model) -> None:
    """Zero every accumulator."""
    for module in getattr(model, "_orig_mod", model).modules():
        if isinstance(module, QuantizationStats):
            module.reset()
