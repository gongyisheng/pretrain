"""Per-site activation statistics, accumulated in-graph over a logging window.

`ActivationStats` is a buffer-holder: it folds tensors in during the forward and
`src.metrics.functional.compute_activation_norm` turns the accumulators into RMS at
read time. Installation lives in `src.metrics.convert`.
"""

import torch


# Read inside the compiled forward, so flipping it specializes the graph. Two states,
# two cached graphs -- the cost the windowed cadence is worth paying.
_RECORDING = [False]


class ActivationStats(torch.nn.Module):
    """One site's running (energy, count) over a window; RMS is sqrt(energy / count)."""

    def __init__(self, site: str, device: torch.device):
        super().__init__()
        self.site = site
        for name in ("energy", "count"):
            self.register_buffer(
                name,
                torch.zeros((), dtype=torch.float32, device=device),
                persistent=False,
            )

    def record(self, tensor: torch.Tensor) -> None:
        """Fold one tensor in, accumulating in fp32 whatever the forward's dtype."""
        values = tensor.detach()
        self.energy.add_(values.pow(2).sum(dtype=torch.float32))
        self.count.add_(values.numel())

    def reset(self) -> None:
        self.energy.zero_()
        self.count.zero_()


def linear_input_hook(module, args):
    if _RECORDING[0]:
        module.act_stats.record(args[0])


def set_activation_monitoring_status(enabled: bool) -> None:
    """Arm or disarm accumulation; a window must span a whole optimizer step."""
    _RECORDING[0] = enabled


def reset_activation_stats(model) -> None:
    """Zero every accumulator."""
    for module in getattr(model, "_orig_mod", model).modules():
        if isinstance(module, ActivationStats):
            module.reset()
