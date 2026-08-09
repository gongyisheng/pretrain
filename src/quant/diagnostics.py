from __future__ import annotations

import contextlib

import torch

from src.quant.linear import QuantizedLinear, quantized_gemm
from src.quant.moe import QuantizedSparseMoEBlock, quantized_grouped_gemm
from src.quant.quantize import dequantize_operand, ragged_scale_blocks
from src.quant.utils import str_to_qmax
from src.utils.metric_utils import compute_quantization_metrics


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


def _replay(record):
    """Re-quantize one site's captured operands through the same functions the
    training path calls, so the measured quantization cannot drift from the
    performed one. Returns {gemm_operand: snapshot}, snapshot None for passthrough.

    The two paths order their wgrad operands differently -- linear contracts
    (gᵀ, x), the grouped path contracts (aᵀ, g) -- so the unpacking differs too.
    """
    cfg = record["cfg"]
    fmt, scaling = cfg.dtype, cfg.scaling
    snapshots = {}

    if record["kind"] == "linear":
        x2d, w, compute_dtype = (
            record["x2d"],
            record["weight"],
            record["compute_dtype"],
        )
        _, snapshots["fwd.act"], snapshots["fwd.weight"] = quantized_gemm(
            x2d, w.t(), fmt["act"], fmt["weight"], compute_dtype, scaling, True
        )
        g = record.get("grad2d")
        if g is None:
            return snapshots
        _, snapshots["dgrad.grad"], snapshots["dgrad.weight"] = quantized_gemm(
            g, w, fmt["grad_input"], fmt["weight"], compute_dtype, scaling, True
        )
        _, snapshots["wgrad.grad"], snapshots["wgrad.act"] = quantized_gemm(
            g.t(), x2d, fmt["grad_weight"], fmt["act"], compute_dtype, scaling, True
        )
        return snapshots

    a, b, offs = record["a"], record["b"], record["offs"]
    out_dtype = a.dtype
    _, snapshots["fwd.act"], snapshots["fwd.weight"] = quantized_grouped_gemm(
        a, b, offs, fmt["act"], fmt["weight"], out_dtype, scaling, True
    )
    g = record.get("grad_out")
    if g is None:
        return snapshots
    _, snapshots["dgrad.grad"], snapshots["dgrad.weight"] = quantized_grouped_gemm(
        g,
        b.transpose(-2, -1).contiguous(),
        offs,
        fmt["grad_input"],
        fmt["weight"],
        out_dtype,
        scaling,
        True,
    )
    _, snapshots["wgrad.act"], snapshots["wgrad.grad"] = quantized_grouped_gemm(
        a.mT, g, offs, fmt["act"], fmt["grad_weight"], out_dtype, scaling, True
    )
    return snapshots


def _snapshot_metrics(snapshot):
    """Metrics for one quantized operand. A wgrad operand's scale rows tile each
    expert, so rebuild that mapping -- indexing it as a global tiling reports
    invented error.
    """
    ragged = None
    if snapshot.offs is not None and snapshot.contract_dim == 0:
        ragged = ragged_scale_blocks(
            snapshot.offs, snapshot.quantized_tensor.shape[0], snapshot.block_size
        )
    dequantized = dequantize_operand(
        snapshot.quantized_tensor,
        snapshot.scale,
        snapshot.contract_dim,
        snapshot.block_size,
        ragged=ragged,
    )
    return compute_quantization_metrics(
        snapshot.source_tensor,
        dequantized,
        snapshot.quantized_tensor,
        str_to_qmax(snapshot.fmt),
        offs=snapshot.offs,
    )


def collect_quantization_diagnostics(model, forward_loss) -> dict[str, float]:
    """Quant health for every quantized site, from one eager fwd/bwd on a fixed batch.

    `forward_loss(model)` returns a scalar loss and owns its own autocast context, so
    the capture below sees the dtypes training quantizes. Runs outside the compiled
    model and discards its gradients, so it observes the quantization the training
    step performs without perturbing it. Returns
    `quant/<metric>/<gemm_operand>/<module>` -> float.
    """
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    grads = {name: p.grad for name, p in model.named_parameters()}
    # the forward runs in train mode, so any buffer it mutates -- today the MoE
    # expert-load counters -- would otherwise leak a val batch into training
    buffers = {name: buffer.clone() for name, buffer in model.named_buffers()}
    for parameter in model.parameters():
        parameter.grad = None
    try:
        with _capture(model) as store:
            forward_loss(model).backward()
        metrics = {}
        for site, record in store.items():
            for gemm_operand, snapshot in _replay(record).items():
                if snapshot is None:  # that operand's format is passthrough
                    continue
                for name, value in _snapshot_metrics(snapshot).items():
                    metrics[f"quant/{name}/{gemm_operand}/{site}"] = value
    finally:
        for name, parameter in model.named_parameters():
            parameter.grad = grads[name]
        for name, buffer in model.named_buffers():
            buffer.copy_(buffers[name])
        torch.random.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
    if not metrics:
        return {}
    keys = list(metrics)
    return dict(zip(keys, torch.stack([metrics[k] for k in keys]).tolist()))  # one sync
