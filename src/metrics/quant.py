"""Per-operand quantization health, folded in where the quantization happens.

`QuantizationStats` holds the running sums a logging window accumulates. The fold
lives here, next to the accumulator it fills; the read is
`src.metrics.functional.compute_quantization_metrics`, which walks the model for
these modules. The two are split because the reported metrics are not averageable
across micro-batches but their operands are: sqnr is a log of a ratio of two sums.
"""

import torch

from src.quant.quantize import dequantize_operand


# Read inside the compiled forward/backward, so flipping it specializes the graph.
# Separate from the activation flag: the two features have independent config flags
# and this fold is far costlier, so their cadences are free to diverge.
_RECORDING = [False]


class QuantizationStats(torch.nn.Module):
    """One GEMM operand's running quantization-error sums over a window.

    `n_groups` is 1 for a plain `QuantizedLinear` operand and the expert count for a
    grouped one, where every metric is reduced per expert before being combined --
    one cold, badly scaled expert has to show up rather than be averaged away by
    whichever experts hold the most tokens.
    """

    FIELDS = ("src_sq", "err_sq", "under", "numel", "nonzero")

    def __init__(self, key: str, n_groups: int, device: torch.device):
        super().__init__()
        # the full metric key, resolved at install time: which GEMM and operand this
        # accumulator belongs to is not a module path, so nothing can re-derive it
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


def accumulate_quantization_sums(source_tensor, codes, dequantized_tensor, offs=None):
    """One operand's quantization-error partial sums, for folding into a window.

    Returns `(src_sq, err_sq, under, numel, nonzero)`, each a 1D fp32 tensor:
    length 1 for a dense operand, length n_experts for a grouped one.

    Two counts, two jobs: `numel` is every element, and stays the validity signal
    that marks an expert as having received tokens at all; `nonzero` counts the
    elements that could underflow, which is what the rate divides by.

    `codes` are the low-precision values themselves, and scales are positive, so
    `codes == 0` is exactly the operand rounding to zero. There is no matching
    saturation count: every scale lands amax at or below qmax by construction, so
    the clamp in `_compute_codes` cannot bind for any amax fp32 can hold.

    With `offs` (cumulative end-offsets) the sums are per expert. One grouped GEMM
    covers every expert and each is quantized against its own amax, so reducing over
    the whole operand would let the experts holding the most tokens set the number --
    one cold, badly scaled expert has to show up. Stacked expert weights carry the
    expert on dim 0; dispatched rows are ragged along `offs`.

    These are sums, not the reported metrics, precisely because the reported metrics
    do not accumulate: sqnr is a log of a ratio. `compute_quantization_metrics`
    divides once, at read time, over the whole window.
    """
    source = source_tensor.float()
    dequantized = dequantized_tensor.float()
    code_values = codes.float()  # fp8 has no abs/eq kernels of its own
    squares = source.square()
    err_squares = (source - dequantized).square()
    nonzero_mask = source != 0
    underflows = (nonzero_mask & (code_values == 0)).float()

    if offs is None:
        src_sq = squares.sum().reshape(1)
        err_sq = err_squares.sum().reshape(1)
        under = underflows.sum().reshape(1)
        # bool mask reduced straight to fp32 -- no full-size float temporary needed
        nonzero = nonzero_mask.sum(dtype=torch.float32).reshape(1)
        numel = torch.full_like(src_sq, source.numel())
    elif source.ndim == 3:  # stacked expert weights, expert on dim 0
        src_sq = squares.flatten(1).sum(1)
        err_sq = err_squares.flatten(1).sum(1)
        under = underflows.flatten(1).sum(1)
        nonzero = nonzero_mask.flatten(1).sum(1, dtype=torch.float32)
        numel = torch.full_like(src_sq, source[0].numel())
    else:  # dispatched rows, ragged along offs -- a row belongs to exactly one expert
        n_groups = offs.shape[0]
        # right=True: row r belongs to the group whose end-offset first exceeds r
        rows = torch.arange(source.shape[0], device=offs.device)
        ids = torch.searchsorted(offs, rows, right=True).clamp_(max=n_groups - 1)

        def by_expert(t):
            per_row = t.flatten(1).sum(1)
            return per_row.new_zeros(n_groups).index_add_(0, ids, per_row)

        src_sq, err_sq, under, nonzero = (
            by_expert(squares),
            by_expert(err_squares),
            by_expert(underflows),
            by_expert(nonzero_mask.float()),  # index_add_ needs a float tensor
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
    scaling,
    offs=None,
    ragged_dim=None,
) -> None:
    """Fold one quantized operand into `stats`, if there is one and it is armed.

    Called from inside `quantized_gemm`/`quantized_grouped_gemm`, immediately after the
    quantize that produced `codes`, so `source` and `codes` are the exact tensors the
    GEMM consumes.

    `offs` reaches both the dequantize and the sums, and the two read it differently on
    purpose: `dequantize_operand` branches on `ragged_dim`, so a `None` ragged_dim still
    takes the dense per-expert path, while the sums branch on `offs` itself to choose a
    per-expert rather than a global reduction. Narrowing `offs` to match `ragged_dim`
    would silently drop grouped weight metrics to one global number.
    """
    if stats is None or not _RECORDING[0]:
        return
    # detach first: `codes`/`scale` are live graph nodes in the forward, so folding
    # them in would make the accumulator buffers require grad and retain the step's
    # graph. Measurement must not join the computation it measures.
    source, codes, scale = source.detach(), codes.detach(), scale.detach()
    dequantized = dequantize_operand(
        codes, scale, contract_dim, scaling, offs=offs, ragged_dim=ragged_dim
    )
    sums = accumulate_quantization_sums(source, codes, dequantized, offs=offs)
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
