"""Install metric accumulators onto a model, mirroring `src.quant.convert`.

Both passes register their accumulators as child modules, so the reset and read
walks find them by `isinstance` without anything having to hold a list.
"""

import torch

from src.metrics.activation import ActivationStats, linear_input_hook
from src.metrics.quant import QuantizationStats
from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock, scaled_grouped_mm_fn
from src.quant.utils import is_quantized


# Which config dtype key each GEMM operand is quantized under. Mirrors the calls in
# QuantizedLinearFn / ScaledGroupedGemmFn: an operand whose format is passthrough is
# never quantized, so it gets no accumulator and the fold is skipped outright.
_QUANT_OPERANDS = {
    "fwd_act": ("fwd.act", "act"),
    "fwd_weight": ("fwd.weight", "weight"),
    "dgrad_grad": ("dgrad.grad", "grad_input"),
    "dgrad_weight": ("dgrad.weight", "weight"),
    "wgrad_grad": ("wgrad.grad", "grad_weight"),
    "wgrad_act": ("wgrad.act", "act"),
}


def apply_activation_monitoring(model) -> None:
    """Attach an ActivationStats child and hook to every Linear; call before compile."""
    # list(): adding children mutates the tree modules() is walking
    for module in list(getattr(model, "_orig_mod", model).modules()):
        if isinstance(module, torch.nn.Linear):
            # device from the weight; a bare torch.zeros(()) takes the process default
            module.add_module("act_stats", ActivationStats(module.weight.device))
            module.register_forward_pre_hook(linear_input_hook)


def _operand_stats(prefix, cfg, n_groups, device):
    """One site's accumulators, skipping operands whose format is passthrough."""
    return {
        lookup: QuantizationStats(f"{label}/{prefix}", n_groups, device)
        for lookup, (label, dtype_key) in _QUANT_OPERANDS.items()
        if is_quantized(cfg.dtype[dtype_key])
    }


def _register(module, sites):
    """Make the accumulators children so the reset/read walks reach them.

    The dict stays the lookup the GEMM uses -- an nn.ModuleDict has no `.get`, and a
    module path cannot express which GEMM and operand a site is anyway.
    """
    module.add_module("_quant_stats", torch.nn.ModuleList(sites))


def apply_quantization_monitoring(model) -> None:
    """Attach quantization accumulators to every quantized site; call before compile.

    Linear sites hold theirs in a dict the autograd Function receives as a non-tensor
    argument, the way `cfg` already travels. MoE sites cannot use a hook -- routing
    happens inside the block's forward -- so their `expert_mm` seam is rebuilt with the
    stats bound into the closure.
    """
    for name, module in getattr(model, "_orig_mod", model).named_modules():
        cfg = getattr(module, "quantization_config", None)
        if isinstance(module, QuantizedLinear):
            module.quant_stats = _operand_stats(name, cfg, 1, module.weight.device)
            _register(module, module.quant_stats.values())
        elif isinstance(module, QuantizedSparseMoEBlock):
            # every expert is quantized against its own amax, so the accumulators are
            # per expert: one cold, badly scaled expert has to show up
            device = next(module.parameters()).device
            by_projection = {
                projection: _operand_stats(
                    f"{name}.expert_{projection}", cfg, module.n_routed_experts, device
                )
                for projection in ("gate_up", "down")
            }
            module.expert_mm = scaled_grouped_mm_fn(cfg, by_projection)
            _register(
                module,
                [s for sites in by_projection.values() for s in sites.values()],
            )
