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
    # SparseMoEBlock allocates the stacked expert weights with torch.empty and relies
    # on TransformerLM._init_weights; a bare block never reaches it, so seeding alone
    # would not make two builds identical.
    nn.init.normal_(block.expert_gate_up, mean=0.0, std=0.02)
    nn.init.normal_(block.expert_down, mean=0.0, std=0.02)
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


# --- replay + entrypoint ---


def _loss(x):
    return lambda model: model(x).square().mean()


@fp8_only
def test_diagnostics_report_every_site_and_metric_for_a_linear():
    torch.manual_seed(0)
    model = _quantized_mlp().cuda().to(torch.bfloat16)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)

    got = diagnostics.collect_quantization_diagnostics(model, _loss(x))

    assert set(got) == {
        f"quant/{metric}/{operand}/down_proj"
        for operand in GEMM_OPERANDS
        for metric in ("sqnr", "underflow_rate", "clip_rate")
    }
    assert all(isinstance(v, float) for v in got.values())


@fp8_only
def test_diagnostics_report_the_moe_projections_with_spread_metrics():
    torch.manual_seed(0)
    model = _quantized_moe().cuda().to(torch.bfloat16)
    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16)

    def forward_loss(m):
        out, _ = m.mlp(x)
        return out.square().mean()

    got = diagnostics.collect_quantization_diagnostics(model, forward_loss)

    assert set(got) == {
        f"quant/{metric}/{operand}/mlp.{projection}"
        for projection in ("expert_gate_up", "expert_down")
        for operand in GEMM_OPERANDS
        for metric in (
            "sqnr",
            "underflow_rate",
            "clip_rate",
            "sqnr_min",
            "underflow_rate_max",
            "clip_rate_max",
        )
    }


@fp8_only
def test_diagnostics_skip_passthrough_operands():
    # grad_weight in bf16 -> the wgrad.grad operand is never quantized.
    torch.manual_seed(0)
    model = _quantized_mlp({"grad_weight": "bf16"}).cuda().to(torch.bfloat16)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)

    got = diagnostics.collect_quantization_diagnostics(model, _loss(x))
    assert not any("wgrad.grad" in key for key in got)
    assert any("dgrad.grad" in key for key in got)


def test_diagnostics_on_a_model_with_no_quantized_sites_are_empty():
    model = nn.Linear(8, 8)
    x = torch.randn(4, 8)
    assert diagnostics.collect_quantization_diagnostics(model, _loss(x)) == {}


@fp8_only
def test_diagnostics_leave_grads_rng_and_seams_untouched():
    torch.manual_seed(0)
    model = _quantized_mlp().cuda().to(torch.bfloat16)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    model(x).sum().backward()
    grad_before = model.down_proj.weight.grad.clone()
    rng_before = torch.random.get_rng_state()

    diagnostics.collect_quantization_diagnostics(model, _loss(x))

    assert torch.equal(model.down_proj.weight.grad, grad_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert not model.down_proj._forward_hooks


@fp8_only
def test_diagnostics_restore_grads_when_the_forward_raises():
    model = _quantized_mlp().cuda().to(torch.bfloat16)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    model(x).sum().backward()
    grad_before = model.down_proj.weight.grad.clone()

    def boom(_model):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        diagnostics.collect_quantization_diagnostics(model, boom)
    assert torch.equal(model.down_proj.weight.grad, grad_before)


# --- parity against the probe this replaces (deleted in the next commit) ---


def _probe_metrics(model, forward_loss):
    from src.quant.convert import attach_quantization_probes
    from src.quant.metrics import QuantizationMetricsCollector

    collector = QuantizationMetricsCollector()
    attach_quantization_probes(model, collector)
    collector.enabled = True
    forward_loss(model).backward()
    collector.enabled = False
    return collector.to_metrics_dict()


@fp8_only
@pytest.mark.parametrize("granularity", ["tensorwise", "rowwise", "blockwise"])
def test_replay_matches_the_probe_for_a_linear(granularity):
    scaling = {"granularity": granularity}
    if granularity == "blockwise":
        scaling |= {"block_size": 32, "scale_dtype": "fp32"}

    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)

    torch.manual_seed(1)
    probed = _quantized_mlp(scaling=scaling).cuda().to(torch.bfloat16)
    expected = _probe_metrics(probed, _loss(x))

    torch.manual_seed(1)
    replayed = _quantized_mlp(scaling=scaling).cuda().to(torch.bfloat16)
    got = diagnostics.collect_quantization_diagnostics(replayed, _loss(x))

    assert set(expected) <= set(got)  # got also carries the new clip_rate
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-5, abs=1e-6), key


@fp8_only
def test_replay_matches_the_probe_for_the_moe_projections():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16)

    def forward_loss(m):
        out, _ = m.mlp(x)
        return out.square().mean()

    torch.manual_seed(1)
    probed = _quantized_moe().cuda().to(torch.bfloat16)
    expected = _probe_metrics(probed, forward_loss)

    torch.manual_seed(1)
    replayed = _quantized_moe().cuda().to(torch.bfloat16)
    got = diagnostics.collect_quantization_diagnostics(replayed, forward_loss)

    assert set(expected) <= set(got)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-5, abs=1e-6), key


def test_replay_matches_the_probe_for_ragged_blockwise_empty_experts():
    """counts=[6, 0, 5]: a wgrad operand whose scale rows tile each expert, with one
    expert holding no tokens -- the case the probe's own regression test pins."""
    from src.quant.metrics import QuantizationMetricProbe
    from src.quant.quantize import QuantizationSnapshot, quantize_operand

    torch.manual_seed(0)
    counts = [6, 0, 5]
    offs = torch.tensor(counts).cumsum(0).to(torch.int32)
    x = torch.randn(sum(counts), 8)
    x[:6] *= 50.0
    blocks = diagnostics.ragged_scale_blocks(offs, x.shape[0], 4)
    xq, scale = quantize_operand(x, 0, "blockwise", 4, "fp8_e4m3", ragged=blocks)
    snapshot = QuantizationSnapshot(x, xq, scale, 0, 4, offs, "fp8_e4m3")

    probe = QuantizationMetricProbe(enabled=True)
    probe.record("wgrad.act", snapshot)

    got = diagnostics._snapshot_metrics(snapshot)
    for key, value in probe.metrics["wgrad.act"].items():
        assert got[key].item() == pytest.approx(value.item(), rel=1e-6), key
    assert got["sqnr"].item() > 15.0
