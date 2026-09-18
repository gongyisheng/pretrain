import torch

# Flipping this re-specializes the compiled graph. Independent of the activation
# flag because this fold is far costlier, so their cadences are free to diverge.
_RECORDING = [False]


class QuantizationStats(torch.nn.Module):
    """Running global quantization-error sums for one GEMM operand over a window."""

    FIELDS = ("src_sq", "err_sq", "under", "numel", "nonzero")

    def __init__(self, key: str, device: torch.device):
        super().__init__()
        # Resolved at install time: which GEMM/operand this belongs to is not a
        # module path, so nothing can re-derive it.
        self.key = key
        for name in self.FIELDS:
            self.register_buffer(
                name,
                torch.zeros(1, dtype=torch.float32, device=device),
                persistent=False,
            )

    def reset(self) -> None:
        for buffer in self.buffers():
            buffer.zero_()


def accumulate_quantization_sums(
    source_tensor: torch.Tensor,
    codes: torch.Tensor,
    dequantized_tensor: torch.Tensor,
    contract_dim: int | None = None,
    rotated_source: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return five global FP32 sums."""
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

    src_sq = squares.sum().reshape(1)
    err_sq = err_squares.sum().reshape(1)
    under = underflows.sum().reshape(1)
    nonzero = nonzero_mask.sum(dtype=torch.float32).reshape(1)
    numel = torch.full_like(src_sq, source.numel())

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
