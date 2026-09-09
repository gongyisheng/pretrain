import pytest
import torch

from src.quant.constants import _FP4_FORMATS, _FP8_FORMATS, _INT8_FORMATS
from src.quant.utils import (
    is_fp4,
    is_fp8,
    resolve_quantization_config,
    scaled_grouped_mm_op,
    scaled_mm_op,
    should_quantize,
    str_to_dtype,
)
from src.utils.config import QuantizationConfig

# --- element formats ---

INT8_FORMATS = sorted(_INT8_FORMATS)
FP8_FORMATS = sorted(_FP8_FORMATS)
FP4_FORMATS = sorted(_FP4_FORMATS)
PASSTHROUGH_FORMATS = ["fp32", "fp16", "bf16"]
RECIPES = ["fp8", "mxfp8", "nvfp4"]

DTYPE_BY_FORMAT = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e8m0": torch.float8_e8m0fnu,
    # E2M1 stores two logical FP4 codes in each uint8 element.
    "fp4_e2m1": torch.uint8,
    **{fmt: torch.int8 for fmt in INT8_FORMATS},
}

# Recipes are not operand dtypes. Nor are compute dtypes, which must reach the
# dtype helpers loudly instead of becoming an unquantized passthrough.
NON_ELEMENT_FORMATS = PASSTHROUGH_FORMATS + RECIPES

# _FP8_FORMATS was derived from the dtype map, so adding the int entries there
# would have silently made is_fp8 true for them.
IS_FP8_BY_FORMAT = {
    **{fmt: True for fmt in FP8_FORMATS},
    **{fmt: False for fmt in FP4_FORMATS + INT8_FORMATS + NON_ELEMENT_FORMATS},
    "fp8_e8m0": False,  # a scale dtype, never an operand dtype
}

IS_FP4_BY_FORMAT = {
    **{fmt: True for fmt in FP4_FORMATS},
    **{fmt: False for fmt in FP8_FORMATS + INT8_FORMATS + NON_ELEMENT_FORMATS},
    "fp8_e8m0": False,  # a scale dtype, never an operand dtype
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


@pytest.mark.parametrize("fmt", IS_FP4_BY_FORMAT)
def test_is_fp4(fmt):
    assert is_fp4(fmt) is IS_FP4_BY_FORMAT[fmt]


# --- module selection ---


def _rule(include=(), exclude=(), enabled=True):
    return QuantizationConfig(
        enabled=enabled,
        dtype={"recipe": "fp8"},
        include=list(include),
        exclude=list(exclude),
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


# --- GEMM op routing ---

_FP32, _E4M3, _E8M0 = (
    torch.float32,
    torch.float8_e4m3fn,
    torch.float8_e8m0fnu,
)

# Strict and extended `mxfp8`/`nvfp4` recipes collapse to their public ops.
GEMM_OP_CASES = [
    ("int8", "int8", _FP32, (0, 0), "int8"),
    ("fp8_e4m3", "fp8_e4m3", _FP32, (1, 32), "fp8"),
    ("fp8_e4m3", "fp8_e4m3", _E8M0, (1, 32), "mxfp8"),
    ("fp8_e5m2", "fp8_e4m3", _E8M0, (1, 32), "mxfp8"),
    ("fp8_e4m3", "fp8_e4m3", _E8M0, (1, 64), "mxfp8"),
    ("fp8_e4m3", "fp8_e4m3", _E8M0, (32, 32), "mxfp8"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (1, 16), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (1, 32), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (1, 64), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (1, 128), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (16, 16), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (32, 32), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (64, 64), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (128, 128), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (0, 0), None),
    ("fp4_e2m1", "fp4_e2m1", _FP32, (1, 16), None),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (16, 32), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (32, 16), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (32, 64), "nvfp4"),
    ("fp4_e2m1", "fp4_e2m1", _E4M3, (1, 256), "nvfp4"),
    ("fp4_e2m1", "int4", _E4M3, (1, 16), None),
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
