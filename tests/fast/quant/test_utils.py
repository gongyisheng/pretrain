import pytest
import torch

from src.quant.constants import _FP8_FORMATS, _INT8_FORMATS
from src.quant.utils import (
    is_fp8,
    layer_idx_from_fqn,
    resolve_quantization_config,
    scaled_grouped_mm_op,
    scaled_mm_op,
    should_quantize,
    str_to_dtype,
)
from src.utils.config import QuantizationConfig

# --- element formats ---

INT_FORMATS = sorted(_INT8_FORMATS)
FP8_FORMATS = sorted(_FP8_FORMATS)
PASSTHROUGH_FORMATS = ["fp32", "fp16", "bf16"]

DTYPE_BY_FORMAT = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e8m0": torch.float8_e8m0fnu,
    **{fmt: torch.int8 for fmt in INT_FORMATS},  # every int width stores in int8
}

# "fp8" expands to fp8_e4m3/fp8_e5m2 per operand, so it is a recipe rather than an
# element format: it must reach the format helpers loudly, not as a silent
# unquantized passthrough. Same for the compute dtypes, which are never elements.
NON_ELEMENT_FORMATS = PASSTHROUGH_FORMATS + ["fp8"]

# _FP8_FORMATS was derived from the dtype map, so adding the int entries there
# would have silently made is_fp8 true for them.
IS_FP8_BY_FORMAT = {
    **{fmt: True for fmt in FP8_FORMATS},
    **{fmt: False for fmt in INT_FORMATS + NON_ELEMENT_FORMATS},
    "fp8_e8m0": False,  # a scale dtype, never an operand format
}


@pytest.mark.parametrize("fmt", DTYPE_BY_FORMAT)
def test_str_to_dtype(fmt):
    assert str_to_dtype(fmt) == DTYPE_BY_FORMAT[fmt]


@pytest.mark.parametrize("fmt", NON_ELEMENT_FORMATS)
def test_str_to_dtype_raise_error(fmt):
    with pytest.raises(KeyError):
        str_to_dtype(fmt)


@pytest.mark.parametrize("fmt", IS_FP8_BY_FORMAT)
def test_is_fp8(fmt):
    assert is_fp8(fmt) is IS_FP8_BY_FORMAT[fmt]


# --- module selection ---


def _rule(include=(), exclude=(), enabled=True, layer_idx=None):
    return QuantizationConfig(
        enabled=enabled,
        dtype={"recipe": "fp8"},
        include=list(include),
        exclude=list(exclude),
        layer_idx=layer_idx,
    )


SHOULD_QUANTIZE_CASES = [
    # (include, exclude, fqn, expected)
    ((), ("lm_head", "*.router"), "lm_head", False),
    ((), ("lm_head", "*.router"), "blocks.0.mlp.router", False),
    ((), ("lm_head",), "blocks.0.attn.q_proj", True),
    ((), (), "blocks.0.attn.q_proj", True),  # no patterns means everything
    ((), ("q_proj",), "blocks.3.attn.q_proj", False),  # leaf name matches anywhere
    ((), ("q_proj",), "blocks.3.attn.k_proj", True),
    (("*.mlp.*",), (), "blocks.0.mlp.down_proj", True),  # allowlist restricts
    (("*.mlp.*",), (), "blocks.0.attn.q_proj", False),
    (("*.mlp.*",), ("*.mlp.gate",), "blocks.0.mlp.down_proj", True),
    (("*.mlp.*",), ("*.mlp.gate",), "blocks.0.mlp.gate", False),  # exclude wins
]


@pytest.mark.parametrize("include,exclude,fqn,expected", SHOULD_QUANTIZE_CASES)
def test_should_quantize(include, exclude, fqn, expected):
    assert should_quantize(fqn, _rule(include, exclude)) is expected


RESOLVE_RULES = [_rule(include=["*.mlp.*"]), _rule(include=["*.attn.*"])]

RESOLVE_CASES = [
    # (rules, fqn, expected index into rules, or None)
    (RESOLVE_RULES, "blocks.0.mlp.down_proj", 0),  # first match wins
    (RESOLVE_RULES, "blocks.0.attn.q_proj", 1),
    (RESOLVE_RULES, "lm_head", None),  # nothing includes it
    ([_rule(include=["*"], enabled=False)], "blocks.0.mlp.down_proj", None),
]


@pytest.mark.parametrize("rules,fqn,expected", RESOLVE_CASES)
def test_resolve_quantization_config(rules, fqn, expected):
    got = resolve_quantization_config(fqn, rules)
    assert got is (None if expected is None else rules[expected])


# --- layer-restricted rules (layer_idx) ---

LAYER_IDX_FROM_FQN_CASES = [
    ("blocks.3.attn.q_proj", 3),
    ("blocks.0.mlp.down_proj", 0),
    ("blocks.12.mlp", 12),
    ("blocks.0", None),  # a bare block is not a quantizable submodule
    ("lm_head", None),  # outside any block
    ("attn.q_proj", None),  # no block prefix (e.g. top-level test model)
    ("embedding", None),
    ("blocks.lm_head.weight", None),  # non-numeric suffix
]


@pytest.mark.parametrize("fqn,expected", LAYER_IDX_FROM_FQN_CASES)
def test_layer_idx_from_fqn(fqn, expected):
    assert layer_idx_from_fqn(fqn) is expected


# (rule layer_idx, fqn, expected) — include/exclude are the rule defaults (all)
SHOULD_QUANTIZE_LAYER_CASES = [
    (None, "blocks.0.attn.q_proj", True),  # no layer_idx means every layer
    (None, "lm_head", True),
    ([0], "blocks.0.attn.q_proj", True),
    ([0, 2], "blocks.2.mlp.down_proj", True),
    ([0, 2], "blocks.1.attn.q_proj", False),
    ([9], "blocks.0.attn.q_proj", False),
    ([0], "lm_head", False),  # outside any block can't claim a layer
    ([1], "blocks.10.attn.q_proj", False),  # exact prefix match, no prefix abuse
    ([10], "blocks.10.attn.q_proj", True),
]


@pytest.mark.parametrize("layer_idx,fqn,expected", SHOULD_QUANTIZE_LAYER_CASES)
def test_should_quantize_layer_idx(layer_idx, fqn, expected):
    assert should_quantize(fqn, _rule(layer_idx=layer_idx)) is expected


def test_should_quantize_layer_idx_composes_with_include():
    rule = _rule(include=["*.attn.*"], layer_idx=[0])
    assert should_quantize("blocks.0.attn.q_proj", rule) is True
    assert should_quantize("blocks.0.mlp.down_proj", rule) is False  # not attn
    assert should_quantize("blocks.1.attn.q_proj", rule) is False  # wrong layer


def test_resolve_quantization_config_layer_rules_first_match_wins():
    layered = _rule(layer_idx=[1])  # a specific layer
    vanilla = _rule()  # a catch-all
    rules = [layered, vanilla]
    assert resolve_quantization_config("blocks.1.attn.q_proj", rules) is layered
    assert resolve_quantization_config("blocks.0.attn.q_proj", rules) is vanilla
    assert resolve_quantization_config("lm_head", rules) is vanilla


def test_resolve_quantization_config_layer_rule_does_not_claim_lm_head():
    # a layer-restricted rule alone must not swallow the head/tied weights
    rules = [_rule(layer_idx=[0])]
    assert resolve_quantization_config("lm_head", rules) is None
    assert resolve_quantization_config("blocks.0.attn.q_proj", rules) is rules[0]


# --- GEMM op routing ---

_FP32, _E8M0 = torch.float32, torch.float8_e8m0fnu

# Both strict `mxfp8` and internal `mxfp8_plus` collapse to the same public op.
GEMM_OP_CASES = [
    ("int8", "int8", _FP32, (0, 0), "int8"),
    ("fp8_e4m3", "fp8_e4m3", _FP32, (1, 32), "fp8"),
    ("fp8_e4m3", "fp8_e4m3", _E8M0, (1, 32), "mxfp8"),
    ("fp8_e5m2", "fp8_e4m3", _E8M0, (1, 32), "mxfp8"),
    ("fp8_e4m3", "fp8_e4m3", _E8M0, (1, 64), "mxfp8"),
    ("fp8_e4m3", "fp8_e4m3", _E8M0, (32, 32), "mxfp8"),
    ("int8", "fp8_e4m3", _FP32, (0, 0), None),
    ("bf16", "bf16", _FP32, (0, 0), None),
]


@pytest.mark.parametrize("a_fmt,b_fmt,scale_dtype,block_shape,family", GEMM_OP_CASES)
def test_scaled_mm_op(a_fmt, b_fmt, scale_dtype, block_shape, family):
    expected = None if family is None else f"gemm.{family}_scaled_mm"
    assert scaled_mm_op(a_fmt, b_fmt, scale_dtype, block_shape) == expected


@pytest.mark.parametrize("a_fmt,b_fmt,scale_dtype,block_shape,family", GEMM_OP_CASES)
def test_scaled_grouped_mm_op(a_fmt, b_fmt, scale_dtype, block_shape, family):
    expected = None if family is None else f"gemm.{family}_scaled_grouped_mm"
    assert scaled_grouped_mm_op(a_fmt, b_fmt, scale_dtype, block_shape) == expected
