import torch

from src.utils.config import QuantConfig
from src.quant.utils import should_quantize, resolve_quantization_config, str_to_dtype


def test_str_to_dtype_maps_fp8_variants():
    assert str_to_dtype("fp8") == torch.float8_e4m3fn
    assert str_to_dtype("fp8_e4m3") == torch.float8_e4m3fn
    assert str_to_dtype("fp8_e5m2") == torch.float8_e5m2
    assert str_to_dtype("fp8_e8m0") == torch.float8_e8m0fnu


def test_str_to_dtype_passthrough_is_none():
    for fmt in ("fp32", "fp16", "bf16"):
        assert str_to_dtype(fmt) is None


def _cfg(**kw):
    base = dict(enabled=True, dtype={"recipe": "fp8"})
    base.update(kw)
    return QuantConfig(**base)


def test_excluded_is_false():
    cfg = _cfg(exclude=["lm_head", "*.router"])
    assert should_quantize("lm_head", cfg) is False
    assert should_quantize("blocks.0.mlp.router", cfg) is False


def test_included_is_true():
    cfg = _cfg(exclude=["lm_head"])
    assert should_quantize("blocks.0.attn.q_proj", cfg) is True


def test_exclude_matches_leaf_name_anywhere():
    cfg = _cfg(exclude=["q_proj"])
    assert should_quantize("blocks.3.attn.q_proj", cfg) is False
    assert should_quantize("blocks.3.attn.k_proj", cfg) is True


def test_include_allowlist_restricts():
    cfg = _cfg(include=["*.mlp.*"], exclude=[])
    assert should_quantize("blocks.0.mlp.down_proj", cfg) is True
    assert should_quantize("blocks.0.attn.q_proj", cfg) is False


def test_empty_include_means_all():
    cfg = _cfg(include=[], exclude=[])
    assert should_quantize("blocks.0.attn.q_proj", cfg) is True


def test_exclude_wins_over_include():
    cfg = _cfg(include=["*.mlp.*"], exclude=["*.mlp.gate"])
    assert should_quantize("blocks.0.mlp.down_proj", cfg) is True
    assert should_quantize("blocks.0.mlp.gate", cfg) is False


def test_resolve_quantization_config_first_match_wins():
    mlp_rule = _cfg(include=["*.mlp.*"], exclude=[])
    attn_rule = _cfg(include=["*.attn.*"], exclude=[])
    configs = [mlp_rule, attn_rule]
    assert resolve_quantization_config("blocks.0.mlp.down_proj", configs) is mlp_rule
    assert resolve_quantization_config("blocks.0.attn.q_proj", configs) is attn_rule
    assert (
        resolve_quantization_config("lm_head", configs) is None
    )  # nothing includes it


def test_resolve_quantization_config_skips_disabled():
    disabled = QuantConfig(enabled=False, include=["*"])
    assert resolve_quantization_config("blocks.0.mlp.down_proj", [disabled]) is None
