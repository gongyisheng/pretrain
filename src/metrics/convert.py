import torch

from src.metrics.activation import ActivationStats, linear_input_hook
from src.metrics.quant import QuantizationStats
from src.quant.constants import GEMM_TENSORS
from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock
from src.quant.utils import is_quantized


def apply_activation_monitoring(model) -> None:
    """Attach activation stats and hooks to every Linear; call before compile."""
    # list() avoids mutation while modules() traverses the tree.
    for module in list(getattr(model, "_orig_mod", model).modules()):
        if isinstance(module, torch.nn.Linear):
            # Match the weight device; default-device allocation may be elsewhere.
            module.add_module("act_stats", ActivationStats(module.weight.device))
            module.register_forward_pre_hook(linear_input_hook)


def _tensor_stats(prefix, cfg, n_groups, device):
    """Create one accumulator per tensor for a site.

    Tensors shared by GEMMs share stats. Passthrough-only tensors get none, so
    their fold is skipped.
    """
    return {
        tensor: QuantizationStats(f"{tensor}/{prefix}", n_groups, device)
        for tensor in GEMM_TENSORS
        if any(is_quantized(fmt) for fmt in cfg.dtype[tensor].values())
    }


def _register(module, sites):
    """Register accumulators as children for reset/read traversal.

    Keep the dict for runtime GEMM lookup: ModuleDict lacks `.get`, while
    ModuleList exposes the modules to traversal.
    """
    module.add_module("_quant_stats", torch.nn.ModuleList(sites))


def apply_quantization_monitoring(model) -> None:
    """Attach quantization stats to every quantized site; call before compile.

    Each site keeps a `quant_stats` dict passed to its autograd Function as a
    non-tensor argument. Module hooks cannot see the operands inside Linear's
    Function or the routing inside a block's forward.
    """
    for name, module in getattr(model, "_orig_mod", model).named_modules():
        cfg = getattr(module, "quantization_config", None)
        if isinstance(module, QuantizedLinear):
            module.quant_stats = _tensor_stats(name, cfg, 1, module.weight.device)
            _register(module, module.quant_stats.values())
        elif isinstance(module, QuantizedSparseMoEBlock):
            # Each expert has its own scale, so stats must expose experts separately.
            device = next(module.parameters()).device
            module.quant_stats = {
                projection: _tensor_stats(
                    f"{name}.expert_{projection}", cfg, module.n_routed_experts, device
                )
                for projection in ("gate_up", "down")
            }
            _register(
                module,
                [s for sites in module.quant_stats.values() for s in sites.values()],
            )
