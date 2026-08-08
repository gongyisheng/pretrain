from __future__ import annotations

import contextlib

import torch

from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock


def _linear_hook(module, name, store):
    """Capture one QuantizedLinear's GEMM operands: the 2D activation, the weight,
    and the output grad -- exactly the reshape and cast QuantizedLinearFn performs.

    compute_dtype is resolved here, inside the caller's autocast region; the backward
    hook below runs outside it and would read the wrong dtype.
    """

    def forward_hook(mod, args, output):
        x = args[0]
        device_type = x.device.type
        compute_dtype = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else x.dtype
        )
        record = {
            "kind": "linear",
            "cfg": mod.quantization_config,
            "compute_dtype": compute_dtype,
            "x2d": x.detach().reshape(-1, x.shape[-1]).to(compute_dtype),
            "weight": mod.weight.detach().to(compute_dtype),
        }
        store[name] = record
        if output.requires_grad:
            output.register_hook(
                lambda g, record=record: record.__setitem__(
                    "grad2d",
                    g.detach().reshape(-1, g.shape[-1]).to(record["compute_dtype"]),
                )
            )

    return module.register_forward_hook(forward_hook)


def _wrap_expert_mm(module, name, store):
    """Wrap the expert_mm attribute to capture the dispatched rows, expert weights and
    offs. A module hook cannot reach them: routing happens inside the block's forward.

    The grad hook rides the post-bias output; addition passes the gradient through
    unchanged, so it is the grad the GEMM's backward receives.
    """
    inner = module.expert_mm

    def wrapped(a, b, offs, projection=None, bias=None):
        out = inner(a, b, offs, projection=projection, bias=bias)
        record = {
            "kind": "moe",
            "cfg": module.quantization_config,
            "a": a.detach(),
            "b": b.detach(),
            "offs": offs,
        }
        store[f"{name}.expert_{projection}"] = record
        if out.requires_grad:
            out.register_hook(
                lambda g, record=record: record.__setitem__("grad_out", g.detach())
            )
        return out

    module.expert_mm = wrapped
    return inner


@contextlib.contextmanager
def _capture(model):
    """Install capture on every quantized site, yield the store, always restore."""
    store = {}
    handles, wrapped = [], []
    try:
        for name, module in model.named_modules():
            if isinstance(module, QuantizedSparseMoEBlock):
                wrapped.append((module, _wrap_expert_mm(module, name, store)))
            elif isinstance(module, QuantizedLinear):
                handles.append(_linear_hook(module, name, store))
        yield store
    finally:
        for handle in handles:
            handle.remove()
        for module, inner in wrapped:
            module.expert_mm = inner
