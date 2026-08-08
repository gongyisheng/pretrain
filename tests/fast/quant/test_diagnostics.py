import pytest
import torch
import torch.nn as nn

from src.layers.mlp import SparseMoEBlock
from src.quant import diagnostics
from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock
from src.utils.config import QuantizationConfig


def _fp8_capable():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="fp8 needs SM >= 8.9")

GEMM_OPERANDS = (
    "fwd.act",
    "fwd.weight",
    "dgrad.grad",
    "dgrad.weight",
    "wgrad.grad",
    "wgrad.act",
)


def _quantized_mlp(fmt_overrides=None, scaling=None):
    cfg = QuantizationConfig(
        enabled=True,
        dtype={"recipe": "fp8", **(fmt_overrides or {})},
        scaling=scaling or {},
    )
    mlp = nn.Sequential()
    mlp.add_module("down_proj", nn.Linear(64, 32, bias=False))
    mlp.down_proj = QuantizedLinear.from_module(mlp.down_proj, cfg)
    return mlp


def _quantized_moe():
    cfg = QuantizationConfig(
        enabled=True, dtype={"recipe": "fp8"}, scaling={"granularity": "rowwise"}
    )
    block = SparseMoEBlock(
        d_model=32,
        intermediate_size=48,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
    )
    parent = nn.Module()
    parent.add_module("mlp", QuantizedSparseMoEBlock.from_module(block, cfg))
    return parent


# --- capture ---


@fp8_only
def test_capture_records_the_linear_operands_in_the_compute_dtype():
    model = _quantized_mlp().cuda()
    x = torch.randn(4, 16, 64, device="cuda")
    with diagnostics._capture(model) as store:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(x).square().mean().backward()

    record = store["down_proj"]
    assert record["kind"] == "linear"
    assert record["compute_dtype"] == torch.bfloat16
    assert record["x2d"].shape == (64, 64)  # flattened to 2D, as the GEMM sees it
    assert record["x2d"].dtype == torch.bfloat16
    assert record["weight"].shape == (32, 64)
    assert record["grad2d"].shape == (64, 32)
    assert record["grad2d"].dtype == torch.bfloat16


@fp8_only
def test_capture_records_both_moe_projections_with_offs():
    model = _quantized_moe().cuda().to(torch.bfloat16)
    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16)
    with diagnostics._capture(model) as store:
        out, _ = model.mlp(x)
        out.sum().backward()

    assert set(store) == {"mlp.expert_gate_up", "mlp.expert_down"}
    for record in store.values():
        assert record["kind"] == "moe"
        assert record["offs"] is not None
        assert record["a"].ndim == 2 and record["b"].ndim == 3
        assert record["grad_out"].shape == record["a"].shape[:1] + record["b"].shape[2:]


@fp8_only
def test_capture_restores_expert_mm_and_removes_hooks():
    model = _quantized_moe().cuda().to(torch.bfloat16)
    linear = _quantized_mlp().cuda()
    before_mm = model.mlp.expert_mm
    with diagnostics._capture(model), diagnostics._capture(linear):
        pass
    assert model.mlp.expert_mm is before_mm
    assert not linear.down_proj._forward_hooks


@fp8_only
def test_capture_restores_even_when_the_forward_raises():
    model = _quantized_moe().cuda().to(torch.bfloat16)
    before_mm = model.mlp.expert_mm
    with pytest.raises(RuntimeError, match="boom"):
        with diagnostics._capture(model):
            raise RuntimeError("boom")
    assert model.mlp.expert_mm is before_mm


@fp8_only
def test_capture_omits_the_grad_when_nothing_requires_it():
    model = _quantized_mlp().cuda()
    x = torch.randn(4, 16, 64, device="cuda")
    with diagnostics._capture(model) as store:
        with torch.no_grad():
            model(x)
    assert "grad2d" not in store["down_proj"]
