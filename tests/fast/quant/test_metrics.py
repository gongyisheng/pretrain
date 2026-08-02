import torch
import torch.nn as nn
import pytest
from src.quant.utils import str_to_min_subnormal, str_to_qmax
from src.quant.quantize import operand_quotient, fake_quantize_operand
from src.quant.metrics import (
    compute_operand_metrics,
    QuantMetricCollector,
    register_quant_metric_hooks,
)
from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock
from src.layers.mlp import SparseMoEBlock
from src.utils.config import QuantConfig


def _fp8_capable():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="fp8 needs SM >= 8.9")


def test_min_subnormal_values():
    assert str_to_min_subnormal("fp8_e4m3") == 2**-9
    assert str_to_min_subnormal("fp8_e5m2") == 2**-16
    assert str_to_min_subnormal("int8") == 0.5
    assert str_to_min_subnormal("int4") == 0.5


def test_min_subnormal_rejects_e8m0():
    with pytest.raises(KeyError):
        str_to_min_subnormal("fp8_e8m0")


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_operand_quotient_dequant_matches_fake_quant(fmt):
    torch.manual_seed(0)
    x = torch.randn(64, 128) * 3.0
    q_abs, deq, xf = operand_quotient(x, -1, "tensorwise", 0, fmt)
    ref = fake_quantize_operand(x, -1, "tensorwise", 0, fmt).float()
    assert torch.allclose(deq, ref, atol=0, rtol=0)
    # amax element maps to exactly qmax under amax scaling
    assert q_abs.max().item() == pytest.approx(str_to_qmax(fmt), rel=1e-4)
    assert xf.shape == x.shape and deq.shape == x.shape


def test_operand_quotient_rowwise_matches_fake_quant():
    torch.manual_seed(1)
    x = torch.randn(32, 96)
    _, deq, _ = operand_quotient(x, -1, "rowwise", 0, "fp8_e4m3")
    ref = fake_quantize_operand(x, -1, "rowwise", 0, "fp8_e4m3").float()
    assert torch.allclose(deq, ref, atol=1e-6)


def test_metrics_passthrough_returns_none():
    x = torch.randn(8, 8)
    assert compute_operand_metrics(x, "bf16", "tensorwise", 0, -1) is None


def test_metrics_underflow_on_outlier_inflated_scale():
    # one huge outlier inflates tensorwise scale so the small bulk underflows.
    x = torch.zeros(1, 256)
    x[0, 0] = 1000.0  # outlier
    x[0, 1:] = 1e-3  # bulk, well below (1000/448)*2^-9 for e4m3
    m = compute_operand_metrics(x, "fp8_e4m3", "tensorwise", 0, -1)
    assert m["underflow_rate"].item() > 0.9  # nearly all bulk flushed
    assert m["range_ratio"].item() > 100.0  # amax/median huge


def test_metrics_clip_zero_under_amax_scaling():
    torch.manual_seed(0)
    x = torch.randn(32, 64)
    m = compute_operand_metrics(x, "fp8_e4m3", "tensorwise", 0, -1)
    # amax maps exactly to qmax; nothing strictly exceeds it.
    assert m["clip_rate"].item() == 0.0


def test_metrics_sqnr_finite_and_positive_for_smooth_tensor():
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    m = compute_operand_metrics(x, "fp8_e4m3", "tensorwise", 0, -1)
    assert torch.isfinite(m["sqnr"]).all()
    assert m["sqnr"].item() > 10.0  # e4m3 on N(0,1) is well above 10 dB


def test_metrics_all_zero_dim_float32():
    x = torch.randn(8, 8)
    m = compute_operand_metrics(x, "fp8_e5m2", "tensorwise", 0, -1)
    for k in ("underflow_rate", "clip_rate", "range_ratio", "sqnr"):
        assert m[k].ndim == 0 and m[k].dtype == torch.float32


def _qlinear(fmt_overrides=None):
    cfg = QuantConfig(enabled=True, dtype={"recipe": "fp8", **(fmt_overrides or {})})
    lin = nn.Linear(64, 32, bias=False)
    q = QuantizedLinear.from_module(lin, cfg)
    q.layer_id = "3"
    return q


@fp8_only
def test_collector_records_and_drains_all_operands():
    torch.manual_seed(0)
    q = _qlinear().cuda().to(torch.bfloat16)
    coll = QuantMetricCollector()
    coll.enabled = True
    handles = register_quant_metric_hooks(q, coll)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    q(x).square().mean().backward()
    out = coll.drain()
    for operand in ("weight", "act", "grad_input", "grad_weight"):
        assert f"quant/underflow_rate/{operand}/layer_3" in out
        assert f"quant/sqnr/{operand}/layer_3" in out
    assert coll.drain() == {}  # cleared after drain
    for h in handles:
        h.remove()


@fp8_only
def test_collector_disabled_records_nothing():
    q = _qlinear().cuda().to(torch.bfloat16)
    coll = QuantMetricCollector()
    coll.enabled = False
    register_quant_metric_hooks(q, coll)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    q(x).square().mean().backward()
    assert coll.drain() == {}


@fp8_only
def test_collector_passthrough_grad_absent():
    # grad_weight forced to bf16 -> no grad_weight series, others present.
    q = _qlinear({"grad_weight": "bf16"}).cuda().to(torch.bfloat16)
    coll = QuantMetricCollector()
    coll.enabled = True
    register_quant_metric_hooks(q, coll)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    q(x).square().mean().backward()
    out = coll.drain()
    assert not any("grad_weight" in k for k in out)
    assert any("grad_input" in k for k in out)


def _moe_block():
    cfg = QuantConfig(
        enabled=True, dtype={"recipe": "fp8"}, scaling={"granularity": "tensorwise"}
    )
    block = SparseMoEBlock(
        d_model=32,
        intermediate_size=48,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
    )
    return QuantizedSparseMoEBlock.from_module(block, cfg, "5")


@fp8_only
def test_collector_records_moe_block():
    # BS = 4 * 16 = 64 tokens, comfortably >= 32 per the fused fp8 wgrad
    # GEMM's contraction-dim floor.
    torch.manual_seed(0)
    block = _moe_block().cuda().to(torch.bfloat16)
    coll = QuantMetricCollector()
    coll.enabled = True
    handles = register_quant_metric_hooks(block, coll)
    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = block(x)
    out.sum().backward()
    out_metrics = coll.drain()
    assert "quant/underflow_rate/weight/layer_5" in out_metrics
    assert "quant/underflow_rate/act/layer_5" in out_metrics
    for h in handles:
        h.remove()


@fp8_only
def test_collector_disabled_records_nothing_moe():
    block = _moe_block().cuda().to(torch.bfloat16)
    coll = QuantMetricCollector()
    coll.enabled = False
    register_quant_metric_hooks(block, coll)
    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = block(x)
    out.sum().backward()
    assert coll.drain() == {}
