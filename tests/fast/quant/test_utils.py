import pytest
import torch

from src.utils.config import QuantizationConfig
from src.quant.utils import (
    is_fp8,
    scaled_grouped_mm_op,
    scaled_mm_op,
    should_quantize,
    resolve_quantization_config,
    str_to_dtype,
)


def test_str_to_dtype_maps_fp8_variants():
    assert str_to_dtype("fp8_e4m3") == torch.float8_e4m3fn
    assert str_to_dtype("fp8_e5m2") == torch.float8_e5m2
    assert str_to_dtype("fp8_e8m0") == torch.float8_e8m0fnu


def test_str_to_dtype_stores_every_int_format_as_int8():
    for fmt in ("int8", "int7", "int6", "int5", "int4"):
        assert str_to_dtype(fmt) == torch.int8


def test_str_to_dtype_rejects_passthrough():
    for fmt in ("fp32", "fp16", "bf16"):
        with pytest.raises(KeyError):
            str_to_dtype(fmt)


def test_int_formats_are_not_fp8():
    # _FP8_FORMATS was derived from the dtype map, so adding the int entries
    # there would have silently made is_fp8 true for them.
    for fmt in ("int8", "int7", "int6", "int5", "int4"):
        assert not is_fp8(fmt)


def test_fp8_is_a_recipe_not_an_element_format():
    # "fp8" expands to fp8_e4m3/fp8_e5m2 per operand, so it must never reach the
    # element-format helpers -- loudly, not as a silent unquantized passthrough.
    with pytest.raises(KeyError):
        str_to_dtype("fp8")
    assert not is_fp8("fp8")


def _cfg(**kw):
    base = dict(enabled=True, dtype={"recipe": "fp8"})
    base.update(kw)
    return QuantizationConfig(**base)


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
    disabled = QuantizationConfig(enabled=False, include=["*"])
    assert resolve_quantization_config("blocks.0.mlp.down_proj", [disabled]) is None


def test_int8_pair_resolves_to_the_int8_op():
    assert scaled_mm_op("int8", "int8", torch.float32, (0, 0)) == "gemm.int8_scaled_mm"


def test_fp8_pair_with_fp32_scales_resolves_to_the_fp8_op():
    assert scaled_mm_op("fp8_e4m3", "fp8_e4m3", torch.float32, (1, 32)) == (
        "gemm.fp8_scaled_mm"
    )


def test_fp8_pair_with_e8m0_scales_resolves_to_the_mxfp8_op():
    assert (
        scaled_mm_op("fp8_e4m3", "fp8_e4m3", torch.float8_e8m0fnu, (1, 32))
        == "gemm.mxfp8_scaled_mm"
    )


def test_wrong_fp8_format_with_mx_scales_resolves_to_the_fp8_op():
    assert (
        scaled_mm_op("fp8_e5m2", "fp8_e4m3", torch.float8_e8m0fnu, (1, 32))
        == "gemm.fp8_scaled_mm"
    )


def test_wrong_block_shape_with_mx_scales_resolves_to_the_fp8_op():
    assert (
        scaled_mm_op("fp8_e4m3", "fp8_e4m3", torch.float8_e8m0fnu, (1, 64))
        == "gemm.fp8_scaled_mm"
    )


def test_grouped_mm_resolves_to_the_grouped_op():
    assert scaled_grouped_mm_op("int8", "int8", torch.float32, (0, 0)) == (
        "gemm.int8_scaled_grouped_mm"
    )


def test_grouped_mm_with_mx_contract_resolves_to_the_mxfp8_op():
    assert (
        scaled_grouped_mm_op("fp8_e4m3", "fp8_e4m3", torch.float8_e8m0fnu, (1, 32))
        == "gemm.mxfp8_scaled_grouped_mm"
    )


def test_mixed_family_pair_has_no_op():
    assert scaled_mm_op("int8", "fp8_e4m3", torch.float32, (0, 0)) is None


def test_unquantized_pair_has_no_op():
    assert scaled_mm_op("bf16", "bf16", torch.float32, (0, 0)) is None
