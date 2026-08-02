from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.quant.constants import _INT8_FORMATS
from src.quant.convert import apply_quantization
from src.quant.linear import QuantLinear
from src.quant.utils import is_supported
from src.utils.config import ModelConfig, QuantConfig, TrainConfig, TrainingConfig


fp8_only = pytest.mark.skipif(not is_supported("fp8"), reason="fp8 needs SM >= 8.9")


class _Tiny(nn.Module):
    def __init__(self, tie=False):
        super().__init__()
        self.token_emb = nn.Embedding(64, 32)
        self.attn = nn.ModuleDict({"q_proj": nn.Linear(32, 32, bias=False)})
        self.mlp = nn.ModuleDict({"down_proj": nn.Linear(32, 32, bias=False)})
        self.lm_head = nn.Linear(32, 64, bias=False)
        if tie:
            self.lm_head.weight = self.token_emb.weight


def _cfg(quant, mixed_precision="bf16"):
    return TrainConfig(
        model=ModelConfig(
            d_model=32,
            n_layers=1,
            vocab_size=64,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 2}}],
        ),
        training=TrainingConfig(mixed_precision=mixed_precision, quant=quant),
    )


def test_disabled_is_noop():
    m = _Tiny()
    apply_quantization(m, _cfg({"enabled": False}))
    assert isinstance(m.attn["q_proj"], nn.Linear)
    assert not isinstance(m.attn["q_proj"], QuantLinear)


@pytest.mark.skipif(
    is_supported("fp8"), reason="testing the unsupported-hardware branch"
)
def test_raises_without_capable_gpu():
    m = _Tiny()
    with pytest.raises(RuntimeError, match="compute capability"):
        apply_quantization(m, _cfg({"enabled": True, "dtype": {"recipe": "fp8"}}))


@pytest.mark.skipif(
    is_supported("fp8"), reason="testing the unsupported-hardware branch"
)
def test_mxfp8_rule_raises_without_capable_gpu():
    # mxfp8 is e4m3 on every operand, so it gates on the same fp8 hardware
    # requirement as the plain "fp8" recipe -- no separate Blackwell-only gate.
    m = _Tiny()
    with pytest.raises(RuntimeError, match="compute capability"):
        apply_quantization(m, _cfg({"enabled": True, "dtype": {"recipe": "mxfp8"}}))


@fp8_only
def test_mxfp8_rule_swaps_on_capable_gpu():
    m = _Tiny().cuda().to(torch.bfloat16)
    apply_quantization(m, _cfg({"enabled": True, "dtype": {"recipe": "mxfp8"}}))
    assert isinstance(m.attn["q_proj"], QuantLinear)
    assert isinstance(m.mlp["down_proj"], QuantLinear)


@fp8_only
def test_swaps_and_excludes_lm_head():
    m = _Tiny().cuda().to(torch.bfloat16)
    apply_quantization(m, _cfg({"enabled": True, "dtype": {"recipe": "fp8"}}))
    assert isinstance(m.attn["q_proj"], QuantLinear)
    assert isinstance(m.mlp["down_proj"], QuantLinear)
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, QuantLinear)
    assert isinstance(m.token_emb, nn.Embedding)


@fp8_only
def test_tie_guard_skips_tied_lm_head():
    m = _Tiny(tie=True).cuda().to(torch.bfloat16)
    # exclude nothing: tie guard alone must keep lm_head unswapped
    cfg = _cfg({"enabled": True, "dtype": {"recipe": "fp8"}, "exclude": []})
    with pytest.warns(UserWarning, match="tied"):
        apply_quantization(m, cfg)
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, QuantLinear)
    assert isinstance(m.attn["q_proj"], QuantLinear)  # non-tied still swapped


@fp8_only
def test_include_restricts_scope():
    m = _Tiny().cuda().to(torch.bfloat16)
    cfg = _cfg(
        {
            "enabled": True,
            "dtype": {"recipe": "fp8"},
            "include": ["*attn*"],
            "exclude": [],
        }
    )
    apply_quantization(m, cfg)
    assert isinstance(m.attn["q_proj"], QuantLinear)
    assert isinstance(m.mlp["down_proj"], nn.Linear)  # not in include


@fp8_only
def test_list_of_rules_first_match_wins():
    m = _Tiny().cuda().to(torch.bfloat16)
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
    assert isinstance(m.attn["q_proj"], QuantLinear)
    assert isinstance(m.mlp["down_proj"], QuantLinear)


def _moe_model():
    import torch.nn as nn

    from src.layers.mlp import SparseMoEBlock

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(64, 32)
            self.moe = SparseMoEBlock(
                d_model=32,
                intermediate_size=48,
                n_routed_experts=4,
                n_routed_experts_per_token=2,
            )
            self.lm_head = nn.Linear(32, 64, bias=False)

    return M()


@fp8_only
def test_moe_experts_get_quantized_expert_mm():
    from src.kernel.gemm import grouped_gemm

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
    assert isinstance(m.moe.expert_gate_up, torch.nn.Parameter)
    # expert_mm was replaced with a quantized callable
    assert m.moe.expert_mm is not grouped_gemm


@fp8_only
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
    assert isinstance(m.moe.router.gate, nn.Linear)
    assert not isinstance(m.moe.router.gate, QuantLinear)


@fp8_only
def test_moe_expert_mm_default_when_excluded():
    from src.kernel.gemm import grouped_gemm

    m = _moe_model()
    apply_quantization(
        m,
        _cfg(
            {
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "scaling": {"recipe": "rowwise"},
                "exclude": ["moe"],
            }
        ),
    )
    assert m.moe.expert_mm is grouped_gemm


# --- int8-family config + converter (migrated from the old test_int8.py; int8
# runs on any GPU, so these are not fp8-gated) ---


@pytest.mark.parametrize("fmt", _INT8_FORMATS)
def test_int8s_recipe_expands(fmt):
    # int8-series is weight-only: uniform quant lacks the dynamic range for
    # activations/gradients, so those stay bf16.
    c = QuantConfig(enabled=True, dtype={"recipe": fmt})
    assert c.dtype == {
        "weight": fmt,
        "act": "bf16",
        "grad_input": "bf16",
        "grad_weight": "bf16",
    }


@pytest.mark.parametrize("fmt", _INT8_FORMATS)
def test_int8s_apply_quantization_swaps(fmt):
    model = nn.Sequential(nn.Linear(64, 64))
    rule = QuantConfig(enabled=True, dtype={"recipe": fmt}, include=["0"])
    cfg = SimpleNamespace(training=SimpleNamespace(quant=[rule]))
    apply_quantization(model, cfg)
    assert isinstance(model[0], QuantLinear)


@fp8_only
def test_apply_quantization_sets_layer_id():
    m = _Tiny().cuda().to(torch.bfloat16)
    apply_quantization(m, _cfg({"enabled": True, "dtype": {"recipe": "fp8"}}))
    qls = [mod for mod in m.modules() if isinstance(mod, QuantLinear)]
    assert qls and all(isinstance(mod.layer_id, str) for mod in qls)
