"""Tests for src.metrics.convert — the install pass that decides which modules get
accumulators, on what device, and that installing does not thrash recompiles.
"""

import pytest
import torch

from src.layers.block import TransformerBlock
from src.layers.mlp import SparseMoEBlock
from src.metrics.activation import set_activation_monitoring_status
from src.metrics.convert import (
    apply_activation_monitoring,
    apply_quantization_monitoring,
)
from src.metrics.functional import compute_activation_norm
from src.metrics.quant import set_quantization_monitoring_status
from src.quant.convert import apply_quantization
from src.quant.moe import QuantizedSparseMoEBlock
from src.utils.config import ModelConfig, TrainConfig, TrainingConfig
from tests.fast.helper import make_attn_mask


D_MODEL = 64


def _block_cfg(n_layers=1):
    return ModelConfig(
        d_model=D_MODEL,
        n_layers=n_layers,
        vocab_size=256,
        attn=[
            {
                "attn_cls": "mha",
                "attn_kwargs": {"n_heads": 4, "attn_implementation": "sdpa"},
            }
        ],
        mlp=[
            {
                "mlp_cls": "dense",
                "mlp_kwargs": {
                    "activation_cls": "swiglu",
                },
            }
        ],
        norm_cls="rmsnorm",
        pos_emb_cls="rope",
        pos_emb_kwargs={},
    )


def _run_block(block, x=None):
    x = torch.randn(2, 16, D_MODEL) if x is None else x
    pos = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
    attn_mask, _ = make_attn_mask("causal", "sdpa", pos, x.dtype)
    return block(x, attn_mask=attn_mask)


# ---------------------------------------------------------------------------
# Sites and naming
# ---------------------------------------------------------------------------


def test_sites_are_exactly_the_linear_inputs():
    """Every site is one Linear's GEMM operand, named by its module path. Anything
    that is not a Linear input -- the residual stream, the SwiGLU product -- is not
    reachable by a hook and is deliberately absent."""
    block = TransformerBlock(_block_cfg(), layer_idx=0)
    apply_activation_monitoring(block)
    set_activation_monitoring_status(True)
    _run_block(block)
    names = set(compute_activation_norm(block))
    assert names == {
        n for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)
    }
    assert "mlp.down_proj" in names


def test_embedding_is_not_a_site_but_tied_lm_head_is():
    """torch.nn.Embedding consumes token ids, not an activation_cls; the tied lm_head still
    contracts a real activation_cls against the same weight."""
    model = torch.nn.Module()
    model.token_emb = torch.nn.Embedding(32, D_MODEL)
    model.lm_head = torch.nn.Linear(D_MODEL, 32, bias=False)
    model.lm_head.weight = model.token_emb.weight
    apply_activation_monitoring(model)
    set_activation_monitoring_status(True)
    model.lm_head(model.token_emb(torch.randint(0, 32, (2, 8))))
    assert set(compute_activation_norm(model)) == {"lm_head"}


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------


def test_compiled_model_records_and_does_not_thrash_recompiles(device):
    """Hooks must be installed before compile, and must not close over per-module
    data -- a per-name closure specializes per site and blows the recompile limit."""
    if device != "cuda":
        pytest.skip("compile parity checked on CUDA")
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    model = torch.nn.Sequential(
        *[torch.nn.Linear(D_MODEL, D_MODEL) for _ in range(6)]
    ).to(device)
    apply_activation_monitoring(model)
    compiled = torch.compile(model)
    x = torch.randn(4, D_MODEL, device=device)
    compiled(x)
    set_activation_monitoring_status(True)
    compiled(x)
    norms = compute_activation_norm(model)
    assert len(norms) == 6
    assert counters["stats"]["unique_graphs"] <= 3, counters["stats"]
    assert not counters["graph_break"], dict(counters["graph_break"])


# ---------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------


def test_buffers_follow_the_model_not_the_ambient_default_device():
    """Accumulators must be created on the module's own weight device. conftest sets
    torch.set_default_device('cuda') autouse, so a bare torch.zeros(()) would take
    the process default instead -- and eager quietly cross-device syncs it, leaving
    torch.compile to be the first thing that fails, on a fake tensor mismatch."""
    block = TransformerBlock(_block_cfg(), layer_idx=0).to("cpu")
    apply_activation_monitoring(block)
    assert block.mlp.down_proj.act_stats.energy.device.type == "cpu"
    assert block.attn.q_proj.act_stats.count.device.type == "cpu"


# ---------------------------------------------------------------------------
# Quantization accumulators: one per tensor, not one per GEMM operand
# ---------------------------------------------------------------------------


def _quantized_linear(dtype):
    """One int8-capable QuantizedLinear with monitoring installed, on CPU."""
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)
    cfg = TrainConfig(
        model=_block_cfg(),
        training=TrainingConfig(
            mixed_precision="no",
            quantization={
                "enabled": True,
                "dtype": dtype,
                "scale": {"granularity": "tensorwise"},
            },
        ),
    )
    apply_quantization(model, cfg)
    apply_quantization_monitoring(model)
    return model.proj


def _fold(module, x):
    set_quantization_monitoring_status(True)
    try:
        module(x).sum().backward()
    finally:
        set_quantization_monitoring_status(False)


def test_one_accumulator_per_quantized_tensor():
    """Keyed by tensor, so a tensor consumed by two GEMMs still gets one site, and a
    tensor left in the compute dtype gets none."""
    proj = _quantized_linear({"weight": "int8", "act": "bf16", "grad_out": "bf16"})
    assert set(proj.quant_stats) == {"weight"}
    assert proj.quant_stats["weight"].key == "weight/proj"


def test_grad_out_site_exists_when_only_one_gemm_quantizes_it():
    proj = _quantized_linear(
        {
            "weight": "bf16",
            "act": "bf16",
            "grad_out": {"dgrad": "int8", "wgrad": "bf16"},
        }
    )
    assert set(proj.quant_stats) == {"grad_out"}


def test_grad_out_site_holds_only_the_quantized_gemms_fold():
    """dgrad quantized, wgrad not: the site sees exactly one operand's elements."""
    proj = _quantized_linear(
        {
            "weight": "bf16",
            "act": "bf16",
            "grad_out": {"dgrad": "int8", "wgrad": "bf16"},
        }
    )
    x = torch.randn(4, D_MODEL, requires_grad=True)
    _fold(proj, x)
    # grad_out is (4, D_MODEL); only the dgrad GEMM folded it
    assert proj.quant_stats["grad_out"].numel.item() == 4 * D_MODEL


def test_both_gemms_pool_into_one_tensor_site():
    """Same tensor quantized in both its GEMMs: the sums add, they do not overwrite."""
    proj = _quantized_linear(
        {
            "weight": "bf16",
            "act": "bf16",
            "grad_out": {"dgrad": "int8", "wgrad": "int8"},
        }
    )
    x = torch.randn(4, D_MODEL, requires_grad=True)
    _fold(proj, x)
    assert proj.quant_stats["grad_out"].numel.item() == 2 * 4 * D_MODEL


def test_moe_monitoring_installs_per_projection_stats():
    """One site per (projection, quantized tensor), reachable by the reset/read walk.

    The seam itself is a method keyed on training mode, so installing monitoring has
    nothing to rebind and cannot strand a block on the wrong GEMM.
    """
    model = torch.nn.Module()
    model.moe = SparseMoEBlock(
        d_model=D_MODEL,
        intermediate_size=2 * D_MODEL,
        n_routed_experts=2,
        n_routed_experts_per_token=1,
    )
    cfg = TrainConfig(
        model=_block_cfg(),
        training=TrainingConfig(
            mixed_precision="no",
            quantization={"enabled": True, "dtype": {"weight": "int8"}},
        ),
    )
    apply_quantization(model, cfg)
    moe = model.moe

    assert isinstance(moe, QuantizedSparseMoEBlock)
    moe.eval()
    assert moe.quant_stats == {}
    apply_quantization_monitoring(model)

    assert set(moe.quant_stats) == {"gate_up", "down"}
    assert {
        site.key for sites in moe.quant_stats.values() for site in sites.values()
    } == {"weight/moe.expert_gate_up", "weight/moe.expert_down"}
    # children, so reset_quantization_stats and the metric read reach them
    assert len(moe._quant_stats) == 2
    # per expert: one cold, badly scaled expert has to be visible on its own
    assert moe.quant_stats["gate_up"]["weight"].numel.shape == (2,)
