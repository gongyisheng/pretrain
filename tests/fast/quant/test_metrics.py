import torch
import torch.nn as nn
import pytest

from src.layers.mlp import SparseMoEBlock
from src.quant.convert import attach_quantization_probes
from src.quant.linear import QuantizedLinear
from src.quant.metrics import QuantizationMetricsCollector
from src.quant.moe import QuantizedSparseMoEBlock
from src.quant.quantize import (
    QuantizationSnapshot,
    dequantize_operand,
    quantize_operand,
)
from src.utils.config import QuantConfig
from src.utils.metric_utils import compute_quantization_metrics


def _fp8_capable():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="fp8 needs SM >= 8.9")

LINEAR_SITES = (
    "fwd.act",
    "fwd.weight",
    "dgrad.grad",
    "dgrad.weight",
    "wgrad.grad",
    "wgrad.act",
)
METRICS = ("sqnr", "underflow_rate", "range_ratio")


# --- compute_quantization_metrics: a pure function of (source, dequantized) ---


def test_identical_tensors_have_infinite_sqnr_and_no_loss():
    x = torch.randn(16, 32)
    m = compute_quantization_metrics(x, x)
    assert m["sqnr"].item() > 200.0  # error clamped to EPS, not zero
    assert m["underflow_rate"].item() == 0.0


def test_underflow_counts_nonzero_source_mapped_to_zero():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    deq = torch.tensor([[1.0, 0.0, 0.0, 4.0]])
    assert compute_quantization_metrics(x, deq)["underflow_rate"].item() == 0.5


def test_underflow_ignores_zeros_already_in_the_source():
    x = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
    deq = torch.tensor([[0.0, 0.0, 1.0, 2.0]])
    assert compute_quantization_metrics(x, deq)["underflow_rate"].item() == 0.0


def test_sqnr_matches_hand_computed_ratio():
    x = torch.tensor([[3.0, 4.0]])  # norm 5
    deq = torch.tensor([[3.0, 3.0]])  # error norm 1
    expected = 20.0 * torch.log10(torch.tensor(5.0))
    assert compute_quantization_metrics(x, deq)["sqnr"].item() == pytest.approx(
        expected.item(), rel=1e-5
    )


def test_range_ratio_is_amax_over_median():
    x = torch.tensor([[1.0, 1.0, 1.0, 100.0]])
    assert compute_quantization_metrics(x, x)["range_ratio"].item() == pytest.approx(
        100.0
    )


def test_metrics_are_zero_dim_float32_and_device_local():
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    m = compute_quantization_metrics(x, x)
    assert set(m) == set(METRICS)
    for value in m.values():
        assert value.ndim == 0 and value.dtype == torch.float32


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2", "int8"])
def test_sqnr_positive_and_finite_over_a_real_round_trip(fmt):
    torch.manual_seed(0)
    x = torch.randn(64, 64)
    q, scale = quantize_operand(x, -1, "tensorwise", 0, fmt)
    m = compute_quantization_metrics(
        x, dequantize_operand(q, scale, -1, "tensorwise", 0)
    )
    assert torch.isfinite(m["sqnr"]) and m["sqnr"].item() > 10.0


def test_outlier_inflated_scale_shows_up_as_underflow():
    # One huge outlier inflates the tensorwise scale so the small bulk underflows.
    x = torch.zeros(1, 256)
    x[0, 0] = 1000.0
    x[0, 1:] = 1e-3  # well below (1000/448) * 2^-9 for e4m3
    q, scale = quantize_operand(x, -1, "tensorwise", 0, "fp8_e4m3")
    m = compute_quantization_metrics(
        x, dequantize_operand(q, scale, -1, "tensorwise", 0)
    )
    assert m["underflow_rate"].item() > 0.9
    assert m["range_ratio"].item() > 100.0


# --- collector ---


def _snapshot(x, fmt="fp8_e4m3"):
    q, scale = quantize_operand(x, -1, "tensorwise", 0, fmt)
    return QuantizationSnapshot(x, q, scale, -1, "tensorwise", 0)


def _collector():
    """`enabled` starts False, as the trainer leaves it between log steps."""
    return QuantizationMetricsCollector()


def test_collector_keys_are_metric_site_module():
    collector = _collector()
    probe = collector.get_probe("blocks.3.mlp.down_proj")
    probe.record("fwd.act", _snapshot(torch.randn(8, 8)))
    out = collector.to_metrics_dict()
    assert set(out) == {
        f"quant/{metric}/fwd.act/blocks.3.mlp.down_proj" for metric in METRICS
    }
    assert all(isinstance(v, float) for v in out.values())


def test_drain_clears_the_store():
    collector = _collector()
    collector.get_probe("m").record("fwd.act", _snapshot(torch.randn(8, 8)))
    assert collector.to_metrics_dict()
    assert collector.to_metrics_dict() == {}


def test_probe_skips_passthrough_operands():
    collector = _collector()
    collector.get_probe("m").record("fwd.act", None)
    assert collector.to_metrics_dict() == {}


def test_probe_is_created_once_and_retained_by_the_collector():
    collector = _collector()
    assert collector.get_probe("m") is collector.get_probe("m")


def test_collector_arms_the_probes_it_owns():
    collector = _collector()
    early = collector.get_probe("m")
    assert early.enabled is False
    collector.enabled = True
    assert early.enabled is True
    # a probe attached while armed starts armed, rather than sitting out a step
    assert collector.get_probe("later").enabled is True


# --- end-to-end through QuantizedLinear ---


def _quantized_mlp(fmt_overrides=None, scaling=None):
    cfg = QuantConfig(
        enabled=True,
        dtype={"recipe": "fp8", **(fmt_overrides or {})},
        scaling=scaling or {},
    )
    mlp = nn.Sequential()
    mlp.add_module("down_proj", nn.Linear(64, 32, bias=False))
    mlp.down_proj = QuantizedLinear.from_module(mlp.down_proj, cfg)
    return mlp


@fp8_only
def test_attach_names_probes_by_module_path():
    model = _quantized_mlp()
    collector = _collector()
    attach_quantization_probes(model, collector)
    assert model.down_proj.quantization_probe is collector.get_probe("down_proj")


@fp8_only
def test_forward_backward_records_every_site():
    torch.manual_seed(0)
    model = _quantized_mlp().cuda().to(torch.bfloat16)
    collector = _collector()
    attach_quantization_probes(model, collector)
    collector.enabled = True

    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    model(x).square().mean().backward()

    out = collector.to_metrics_dict()
    assert set(out) == {
        f"quant/{metric}/{site}/down_proj"
        for site in LINEAR_SITES
        for metric in METRICS
    }


@fp8_only
def test_weight_sites_differ_under_rowwise():
    # The weight contracts over K in fwd and over N in dgrad, so under rowwise the
    # two are quantized with different scale partitions and report different sqnr.
    # (Under tensorwise a single whole-tensor amax makes the axis irrelevant.)
    torch.manual_seed(0)
    model = _quantized_mlp(scaling={"granularity": "rowwise"}).cuda().to(torch.bfloat16)
    collector = _collector()
    attach_quantization_probes(model, collector)
    collector.enabled = True

    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    model(x).square().mean().backward()

    out = collector.to_metrics_dict()
    assert (
        out["quant/sqnr/fwd.weight/down_proj"]
        != out["quant/sqnr/dgrad.weight/down_proj"]
    )


@fp8_only
def test_disabled_collector_records_nothing():
    model = _quantized_mlp().cuda().to(torch.bfloat16)
    collector = _collector()
    attach_quantization_probes(model, collector)  # enabled stays False

    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    model(x).square().mean().backward()
    assert collector.to_metrics_dict() == {}


@fp8_only
def test_unattached_model_records_nothing():
    model = _quantized_mlp().cuda().to(torch.bfloat16)
    collector = _collector()
    collector.enabled = True  # no attach_quantization_probes call

    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    model(x).square().mean().backward()
    assert collector.to_metrics_dict() == {}


@fp8_only
def test_passthrough_operand_has_no_series():
    # grad_weight in bf16 -> the wgrad.grad operand is never quantized.
    model = _quantized_mlp({"grad_weight": "bf16"}).cuda().to(torch.bfloat16)
    collector = _collector()
    attach_quantization_probes(model, collector)
    collector.enabled = True

    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    model(x).square().mean().backward()

    out = collector.to_metrics_dict()
    assert not any("wgrad.grad" in key for key in out)
    assert any("dgrad.grad" in key for key in out)


# --- end-to-end through QuantizedSparseMoEBlock ---


def _quantized_moe():
    cfg = QuantConfig(
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


@fp8_only
def test_moe_records_both_projections_separately():
    # BS = 4 * 16 = 64 tokens, comfortably above the fused fp8 wgrad GEMM's
    # contraction-dim floor.
    torch.manual_seed(0)
    model = _quantized_moe().cuda().to(torch.bfloat16)
    collector = _collector()
    attach_quantization_probes(model, collector)
    collector.enabled = True

    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = model.mlp(x)
    out.sum().backward()

    got = collector.to_metrics_dict()
    assert set(got) == {
        f"quant/{metric}/{site}/mlp.{projection}"
        for projection in ("expert_gate_up", "expert_down")
        for site in LINEAR_SITES
        for metric in METRICS
    }


@fp8_only
def test_moe_down_projection_sees_the_post_activation_tensor():
    # The two projections consume different tensors (block input vs post-SwiGLU h),
    # so their activation range must differ -- this is what a single shared
    # expert_mm would have collapsed into one overwritten series.
    torch.manual_seed(0)
    model = _quantized_moe().cuda().to(torch.bfloat16)
    collector = _collector()
    attach_quantization_probes(model, collector)
    collector.enabled = True

    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = model.mlp(x)
    out.sum().backward()

    got = collector.to_metrics_dict()
    assert (
        got["quant/range_ratio/fwd.act/mlp.expert_gate_up"]
        != got["quant/range_ratio/fwd.act/mlp.expert_down"]
    )


@fp8_only
def test_moe_disabled_collector_records_nothing():
    model = _quantized_moe().cuda().to(torch.bfloat16)
    collector = _collector()
    attach_quantization_probes(model, collector)  # enabled stays False

    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = model.mlp(x)
    out.sum().backward()
    assert collector.to_metrics_dict() == {}
