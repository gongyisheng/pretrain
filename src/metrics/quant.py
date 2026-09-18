import torch

from src.kernel.ops import unpack_e2m1
from src.quant.quantize import dequantize_operand, quantize_operand
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
        self.register_buffer(
            "quantization_stats",
            torch.zeros(
                (n_groups, len(self.FIELDS)), dtype=torch.float32, device=device
            ),
            persistent=False,
        )

    def reset(self) -> None:
        self.quantization_stats.zero_()

    @property
    def src_sq(self) -> torch.Tensor:
        return self.quantization_stats[:, 0]

    @src_sq.setter
    def src_sq(self, value: torch.Tensor) -> None:
        self.quantization_stats[:, 0].copy_(value)

    @property
    def err_sq(self) -> torch.Tensor:
        return self.quantization_stats[:, 1]

    @err_sq.setter
    def err_sq(self, value: torch.Tensor) -> None:
        self.quantization_stats[:, 1].copy_(value)

    @property
    def under(self) -> torch.Tensor:
        return self.quantization_stats[:, 2]

    @under.setter
    def under(self, value: torch.Tensor) -> None:
        self.quantization_stats[:, 2].copy_(value)

    @property
    def numel(self) -> torch.Tensor:
        return self.quantization_stats[:, 3]

    @numel.setter
    def numel(self, value: torch.Tensor) -> None:
        self.quantization_stats[:, 3].copy_(value)

    @property
    def nonzero(self) -> torch.Tensor:
        return self.quantization_stats[:, 4]

    @nonzero.setter
    def nonzero(self, value: torch.Tensor) -> None:
        self.quantization_stats[:, 4].copy_(value)


def accumulate_quantization_sums(
    source_tensor,
    codes,
    dequantized_tensor,
    offs=None,
    ragged_dim=None,
    contract_dim=None,
    rotated_source=None,
):
    """Return per-group quantization-error sums for a monitoring window."""
    source = source_tensor.float()
    mask_source = source if rotated_source is None else rotated_source.float()
    dequantized = dequantized_tensor.float()
    if codes.dtype is torch.uint8:
        if contract_dim is None:
            raise ValueError("packed fp4 codes require contract_dim")
        code_values = unpack_e2m1(codes, contract_dim)
    else:
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

        def by_expert(t):
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
    stats,
    source,
    codes,
    scale,
    contract_dim,
    scale_cfg,
    offs=None,
    ragged_dim=None,
    rotation: Rotation | None = None,
    global_scale=None,
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
    if global_scale is not None:
        # Without it the reported error is the operand's whole magnitude, not the
        # quantizer's: the GEMM sees codes scaled by it, so the metric must too.
        global_scale = global_scale.detach()
    # Must match `quantize_operand`'s rotation exactly, or the error metrics skew.
    rotated_source = (
        None if rotation is None else rotation(source, contract_dim, torch.float32)
    )
    dequantized = dequantize_operand(
        codes,
        scale,
        contract_dim,
        scale_cfg,
        offs=offs if ragged_dim is not None else None,
        ragged_dim=ragged_dim,
        rotation=rotation,
        global_scale=global_scale,
    )
    sums = accumulate_quantization_sums(
        source,
        codes,
        dequantized,
        offs=offs,
        ragged_dim=ragged_dim,
        contract_dim=contract_dim,
        rotated_source=rotated_source,
    )
    stats.quantization_stats.add_(torch.stack(sums, dim=-1))


def quantize_and_record(
    stats: QuantizationStats | None,
    source: torch.Tensor,
    contract_dim: int,
    fmt: str,
    scale_cfg: dict,
    stochastic_rounding: bool = False,
    rotation: Rotation | None = None,
    output_layout: str = "row_major",
    backend: str | None = None,
    scale_layout: str = "row_major",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize an operand and record fused statistics when the backend supplies them."""
    collect = (
        stats is not None and _RECORDING[0] and rotation is None and source.ndim == 2
    )
    if scale_layout != "row_major" and (
        backend != "cuda" or rotation is not None or source.ndim != 2
    ):
        raise ValueError("packed scale monitoring requires fused CUDA statistics")
    result = quantize_operand(
        source,
        contract_dim,
        fmt,
        scale_cfg,
        stochastic_rounding=stochastic_rounding,
        rotation=rotation,
        return_quantization_stats=collect,
        output_layout=output_layout,
        backend=backend,
        scale_layout=scale_layout,
    )
    if collect:
        codes, scale, global_scale, quantization_stats = result
        if quantization_stats is not None:
            stats.quantization_stats.add_(quantization_stats)
            return codes, scale, global_scale
    else:
        codes, scale, global_scale = result
    record_operand(
        stats,
        source,
        codes,
        scale,
        contract_dim,
        scale_cfg,
        rotation=rotation,
        global_scale=global_scale,
    )
    return codes, scale, global_scale


def set_quantization_monitoring_status(enabled: bool) -> None:
    """Arm or disarm accumulation; a window must span a whole optimizer step."""
    _RECORDING[0] = enabled


def reset_quantization_stats(model) -> None:
    """Zero every accumulator."""
    for module in getattr(model, "_orig_mod", model).modules():
        if isinstance(module, QuantizationStats):
            module.reset()
