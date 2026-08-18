from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.quant.constants import _INT8_FORMATS
from src.quant.convert import apply_quantization
from src.quant.linear import QuantizedLinear
from src.quant import moe as quant_moe
from src.quant.moe import QuantizedSparseMoEBlock
from src.utils.config import (
    ModelConfig,
    QuantizationConfig,
    TrainConfig,
    TrainingConfig,
)


class _Tiny(nn.Module):
    def __init__(self, tie=False):
        super().__init__()
        self.token_emb = nn.Embedding(64, 32)
        self.attn = nn.ModuleDict({"q_proj": nn.Linear(32, 32, bias=False)})
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(32, 32, bias=False)})
        self.lm_head = nn.Linear(32, 64, bias=False)
        if tie:
            self.lm_head.weight = self.token_emb.weight


def _cfg(quantization, mixed_precision="bf16"):
    return TrainConfig(
        model=ModelConfig(
            d_model=32,
            n_layers=1,
            vocab_size=64,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 2}}],
        ),
        training=TrainingConfig(
            mixed_precision=mixed_precision, quantization=quantization
        ),
    )


def test_disabled_is_noop():
    m = _Tiny()
    apply_quantization(m, _cfg({"enabled": False}))
    assert isinstance(m.attn["q_proj"], nn.Linear)
    assert not isinstance(m.attn["q_proj"], QuantizedLinear)


def test_fp8_rule_swaps_without_hardware_preflight():
    m = _Tiny()
    apply_quantization(m, _cfg({"enabled": True, "dtype": {"recipe": "fp8"}}))
    assert isinstance(m.attn["q_proj"], QuantizedLinear)
    assert isinstance(m.mlp["down_proj"], QuantizedLinear)


def test_mxfp8_rule_swaps_without_hardware_preflight():
    m = _Tiny()
    apply_quantization(m, _cfg({"enabled": True, "dtype": {"recipe": "mxfp8"}}))
    assert isinstance(m.attn["q_proj"], QuantizedLinear)
    assert isinstance(m.mlp["down_proj"], QuantizedLinear)


def test_swaps_and_excludes_lm_head():
    m = _Tiny()
    apply_quantization(m, _cfg({"enabled": True, "dtype": {"recipe": "fp8"}}))
    assert isinstance(m.attn["q_proj"], QuantizedLinear)
    assert isinstance(m.mlp["down_proj"], QuantizedLinear)
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(
        m.lm_head, QuantizedLinear
    )
    assert isinstance(m.token_emb, nn.Embedding)


def test_tie_guard_skips_tied_lm_head(capsys):
    m = _Tiny(tie=True)
    # exclude nothing: tie guard alone must keep lm_head unswapped
    cfg = _cfg({"enabled": True, "dtype": {"recipe": "fp8"}, "exclude": []})
    apply_quantization(m, cfg)
    assert "tied" in capsys.readouterr().out
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(
        m.lm_head, QuantizedLinear
    )
    assert isinstance(m.attn["q_proj"], QuantizedLinear)  # non-tied still swapped


def test_include_restricts_scope():
    m = _Tiny()
    cfg = _cfg(
        {
            "enabled": True,
            "dtype": {"recipe": "fp8"},
            "include": ["*attn*"],
            "exclude": [],
        }
    )
    apply_quantization(m, cfg)
    assert isinstance(m.attn["q_proj"], QuantizedLinear)
    assert isinstance(m.mlp["down_proj"], nn.Linear)  # not in include


def test_list_of_rules_first_match_wins():
    m = _Tiny()
    cfg = _cfg(
        [
            {
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "include": ["*attn*"],
                "exclude": [],
            },
            {
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "include": ["*mlp*"],
                "exclude": [],
            },
        ]
    )
    apply_quantization(m, cfg)
    assert isinstance(m.attn["q_proj"], QuantizedLinear)
    assert isinstance(m.mlp["down_proj"], QuantizedLinear)


def _moe_model():
    import torch.nn as nn

    from src.layers.mlp import SparseMoEBlock

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(64, 32)
            self.mlp = SparseMoEBlock(
                d_model=32,
                intermediate_size=48,
                n_routed_experts=4,
                n_routed_experts_per_token=2,
            )
            self.lm_head = nn.Linear(32, 64, bias=False)

    return M()


def test_moe_experts_get_quantized_expert_mm():
    m = _moe_model()
    apply_quantization(
        m,
        _cfg(
            {
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "scaling": {"recipe": "rowwise"},
            }
        ),
    )
    # routed experts stay Parameters (no module swap)
    assert isinstance(m.mlp.expert_gate_up, torch.nn.Parameter)
    # the block was swapped, so its seam is the quantizing override
    assert m.mlp.expert_mm.__func__ is QuantizedSparseMoEBlock.expert_mm


def test_quantized_moe_uses_plain_expert_gemm_in_eval(monkeypatch):
    from src.layers.mlp import SparseMoEBlock

    with torch.device("cpu"):
        source = SparseMoEBlock(
            d_model=4,
            intermediate_size=8,
            n_routed_experts=2,
            n_routed_experts_per_token=1,
            aux_loss=False,
        )
    source.eval()
    block = QuantizedSparseMoEBlock.from_module(
        source, QuantizationConfig(enabled=True, dtype={"weight": "int4"})
    )

    # Which GEMM the seam dispatches to is the whole question, so spy on both: the
    # base block's seam is what eval falls through to.
    used = []
    monkeypatch.setattr(
        SparseMoEBlock, "expert_mm", lambda *a, **kw: used.append("plain")
    )
    monkeypatch.setattr(
        quant_moe.ScaledGroupedGemmFn,
        "apply",
        staticmethod(lambda *a: used.append("quantized")),
    )
    a = torch.randn(4, 4)
    b = torch.randn(2, 4, 8)
    offs = torch.tensor([2, 4])

    assert not block.training
    block.expert_mm(a, b, offs, projection="gate_up")
    block.train()
    block.expert_mm(a, b, offs, projection="gate_up")
    block.eval()
    block.expert_mm(a, b, offs, projection="gate_up")

    assert used == ["plain", "quantized", "plain"]


def test_router_gate_stays_fp32_linear():
    m = _moe_model()
    apply_quantization(
        m,
        _cfg(
            {
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "scaling": {"recipe": "rowwise"},
            }
        ),
    )
    # gate must NOT be swapped (it is fp32-pinned by MoERouter)
    assert isinstance(m.mlp.router.gate, nn.Linear)
    assert not isinstance(m.mlp.router.gate, QuantizedLinear)


def test_moe_expert_mm_default_when_excluded():
    from src.layers.mlp import SparseMoEBlock

    m = _moe_model()
    apply_quantization(
        m,
        _cfg(
            {
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "scaling": {"recipe": "rowwise"},
                "exclude": ["mlp"],
            }
        ),
    )
    # never converted, so it keeps the base block's unquantized seam
    assert type(m.mlp) is SparseMoEBlock
    assert m.mlp.expert_mm.__func__ is SparseMoEBlock.expert_mm


# --- int8-family config + converter (migrated from the old test_int8.py; int8
# runs on any GPU, so these are not fp8-gated) ---


@pytest.mark.parametrize("fmt", sorted(_INT8_FORMATS))
def test_int8s_recipe_expands(fmt):
    # int8-series is weight-only: uniform quant lacks the dynamic range for
    # activations/gradients, so those stay bf16.
    c = QuantizationConfig(enabled=True, dtype={"recipe": fmt})
    assert c.dtype == {
        "weight": {"fwd": fmt, "dgrad": fmt},
        "act": {"fwd": "bf16", "wgrad": "bf16"},
        "grad_out": {"dgrad": "bf16", "wgrad": "bf16"},
    }


@pytest.mark.parametrize("fmt", sorted(_INT8_FORMATS))
def test_int8s_apply_quantization_swaps(fmt):
    model = nn.Sequential(nn.Linear(64, 64))
    rule = QuantizationConfig(enabled=True, dtype={"recipe": fmt}, include=["0"])
    cfg = SimpleNamespace(training=SimpleNamespace(quantization=[rule]))
    apply_quantization(model, cfg)
    assert isinstance(model[0], QuantizedLinear)


def test_apply_quantization_attaches_no_metric_state():
    # Metrics are collected separately (collect_quantization_diagnostics), so conversion
    # alone must leave no metric state on the swapped modules — nothing in the graph.
    m = _Tiny()
    apply_quantization(m, _cfg({"enabled": True, "dtype": {"recipe": "fp8"}}))
    qls = [mod for mod in m.modules() if isinstance(mod, QuantizedLinear)]
    assert qls and all(not hasattr(mod, "quantization_probe") for mod in qls)
