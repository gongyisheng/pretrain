"""Install metric accumulators onto a model, mirroring `src.quant.convert`.

Both passes register their accumulators as child modules, so the reset and read
walks find them by `isinstance` without anything having to hold a list.
"""

import torch

from src.metrics.activation import ActivationStats, linear_input_hook
from src.metrics.quant import QuantizationStats
from src.quant.constants import QUANT_TENSORS
from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock
from src.quant.utils import is_quantized


def apply_activation_monitoring(model) -> None:
    """Attach an ActivationStats child and hook to every Linear; call before compile."""
    # list(): adding children mutates the tree modules() is walking
    for module in list(getattr(model, "_orig_mod", model).modules()):
        if isinstance(module, torch.nn.Linear):
            # device from the weight; a bare torch.zeros(()) takes the process default
            module.add_module("act_stats", ActivationStats(module.weight.device))
            module.register_forward_pre_hook(linear_input_hook)


def _tensor_stats(prefix, cfg, n_groups, device):
    """One site's accumulators, one per tensor rather than per GEMM operand.

    A tensor consumed by two GEMMs shares an accumulator across both, so the sums
    pool; a tensor left in a passthrough format in every GEMM that consumes it is
    never quantized, so it gets none and the fold is skipped outright.
    """
    return {
        tensor: QuantizationStats(f"{tensor}/{prefix}", n_groups, device)
        for tensor in QUANT_TENSORS
        if any(is_quantized(fmt) for fmt in cfg.dtype[tensor].values())
    }


def _register(module, sites):
    """Make the accumulators children so the reset/read walks reach them.

    The dict stays the lookup the GEMM uses -- an nn.ModuleDict has no `.get`, and a
    module path cannot express which tensor a site is anyway.
    """
    module.add_module("_quant_stats", torch.nn.ModuleList(sites))


def apply_quantization_monitoring(model) -> None:
    """Attach quantization accumulators to every quantized site; call before compile.

    Both kinds hold theirs in a `quant_stats` dict the autograd Function receives as a
    non-tensor argument, the way `cfg` already travels. Neither can use a module hook:
    a Linear's operands are inside its Function, and a block's routing happens inside
    its forward, so nothing a hook sees names the operands.
    """
    for name, module in getattr(model, "_orig_mod", model).named_modules():
        cfg = getattr(module, "quantization_config", None)
        if isinstance(module, QuantizedLinear):
            module.quant_stats = _tensor_stats(name, cfg, 1, module.weight.device)
            _register(module, module.quant_stats.values())
        elif isinstance(module, QuantizedSparseMoEBlock):
            # every expert is quantized against its own amax, so the accumulators are
            # per expert: one cold, badly scaled expert has to show up
            device = next(module.parameters()).device
            projections = ("gate", "up", "down") if module.gated else ("up", "down")
            module.quant_stats = {
                projection: _tensor_stats(
                    f"{name}.expert_{projection}", cfg, module.n_routed_experts, device
                )
                for projection in projections
            }
            _register(
                module,
                [s for sites in module.quant_stats.values() for s in sites.values()],
            )
