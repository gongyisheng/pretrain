import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from src.quant.rotation import (
    HadamardRotation,
    build_rotation,
    build_rotation_key,
)


CONTRACT_DIMS = (-2, -1)
ROTATION_SHAPES = ((64, 128), (3, 64, 128))
GEMM_SHAPES = ((7, 8, 13), (16, 32, 24), (33, 64, 15))
GEMM_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# Worst errors over the full shape/dtype grid are 1.14e-5, 1.56e-2, and 1.25e-1;
# the margins are 4.37x, 4.48x, and 4.00x.
GEMM_ATOL = {
    torch.float32: 5e-5,
    torch.float16: 7e-2,
    torch.bfloat16: 5e-1,
}
# Worst inverse and isometry errors over ROTATION_SHAPES x CONTRACT_DIMS are
# 4.77e-7 and 1.95e-3; the margins are 4.19x and 3.58x.
INVERSE_ATOL = 2e-6
ISOMETRY_ATOL = 7e-3
ROTATION_DEVICES = ["cpu", "cuda"]
INVALID_HADAMARD_KWARGS = [
    {"random_sign": 1},
    {"seed": -1},
    {"seed": True},
    {"block_size": 4, "sign_vector": [1.0, -1.0, 0.0, 1.0]},
    {"unknown": 1},
]
INVALID_HADAMARD_APPLY_CASES = [
    (0, (8, 16), -1, "power of two"),
    (3, (8, 16), -1, "power of two"),
    (1.5, (8, 16), -1, "power of two"),
    (True, (8, 16), -1, "power of two"),
    (8, (8, 10), -1, "block 8 must divide"),
    (8, (8, 16), 0, "contract_dim must be -2 or -1"),
]


def test_hadamard_rotation_is_a_module_with_movable_buffers():
    """Catch a transform that cannot live once at the model root and follow devices."""
    rotation = HadamardRotation(block_size=8, random_sign=True)

    assert isinstance(rotation, nn.Module)
    # The butterfly derives the transform from block_size, so no matrix is held.
    assert dict(rotation.named_buffers()) == {"sign_vector": rotation.sign_vector}
    assert set(rotation.state_dict()) == {"sign_vector"}
    rotation.to("meta")
    assert rotation.sign_vector.device.type == "meta"


@pytest.mark.parametrize("kwargs", INVALID_HADAMARD_KWARGS)
def test_hadamard_rotation_init_raise_error(kwargs):
    error_type = TypeError if "unknown" in kwargs else ValueError
    with pytest.raises(error_type):
        HadamardRotation(**kwargs)


def test_build_rotation_raise_error():
    with pytest.raises(ValueError, match="invalid quant rotation"):
        build_rotation(
            {
                "rotation_cls": "hadamard",
                "rotation_kwargs": {"unknown": 1},
            }
        )


@pytest.mark.parametrize("shape", ROTATION_SHAPES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
def test_hadamard_rotation_inverse(shape, contract_dim):
    torch.manual_seed(0)
    rotation = HadamardRotation(block_size=8, seed=7)
    x = torch.randn(shape)

    transformed = rotation(x, contract_dim)
    restored = rotation.inverse(transformed, contract_dim)

    assert transformed.shape == x.shape
    torch.testing.assert_close(restored, x, atol=INVERSE_ATOL, rtol=0)
    torch.testing.assert_close(
        transformed.square().sum(), x.square().sum(), atol=ISOMETRY_ATOL, rtol=0
    )


@pytest.mark.parametrize("shape", GEMM_SHAPES)
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_hadamard_rotation_apply_gemm(shape, dtype):
    torch.manual_seed(0)
    rotation = HadamardRotation(block_size=8, seed=11)
    m, k, n = shape
    a = torch.randn(m, k, dtype=dtype)
    b = torch.randn(k, n, dtype=dtype)

    transformed = rotation(a, -1) @ rotation(b, -2)

    torch.testing.assert_close(transformed, a @ b, atol=GEMM_ATOL[dtype], rtol=0)


def test_hadamard_rotation_apply_seed():
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    first = HadamardRotation(block_size=8, seed=1234)
    same = HadamardRotation(block_size=8, seed=1234)
    different = HadamardRotation(block_size=8, seed=4321)
    deterministic = HadamardRotation(block_size=8, random_sign=False)

    assert torch.equal(first(x, -1), same(x, -1))
    assert not torch.equal(first(x, -1), different(x, -1))
    assert not torch.equal(first(x, -1), deterministic(x, -1))


@pytest.mark.parametrize("device", ROTATION_DEVICES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_hadamard_rotation_apply_out_dtype(dtype, contract_dim, device):
    """Promoting on the store is exactly the pre-cast it replaces, in one pass."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    rotation = HadamardRotation(block_size=8, seed=7)
    x = torch.randn(64, 128, dtype=dtype, device=device)
    other_device = "cuda" if device == "cpu" else "cpu"

    with torch.device(other_device):
        transformed = rotation(x, contract_dim)
        promoted = rotation(x, contract_dim, torch.float32)

    assert transformed.dtype == dtype
    assert transformed.device == x.device
    assert promoted.dtype is torch.float32
    assert promoted.device == x.device
    assert torch.equal(promoted, rotation(x.float(), contract_dim))
    assert torch.equal(
        rotation.inverse(x, contract_dim, torch.float32),
        rotation.inverse(x.float(), contract_dim),
    )


def test_hadamard_rotation_apply_block_one_identity():
    rotation = HadamardRotation(block_size=1)
    x = torch.randn(3, 4)

    assert rotation(x, -1) is x
    assert torch.equal(rotation.inverse(x, -2), x)
    assert rotation(x, -1, torch.bfloat16).dtype is torch.bfloat16


@pytest.mark.parametrize(
    "block_size, shape, contract_dim, error_match", INVALID_HADAMARD_APPLY_CASES
)
def test_hadamard_rotation_apply_raise_error(
    block_size, shape, contract_dim, error_match
):
    rotation = HadamardRotation(block_size=block_size, random_sign=False)

    with pytest.raises(ValueError, match=error_match):
        rotation(torch.randn(shape), contract_dim)


# --- rotation identity -------------------------------------------------------

BASE_ROTATION = {
    "rotation_cls": "hadamard",
    "rotation_kwargs": {"block_size": 16, "seed": 1},
    "gemms": ["fwd"],
}
BASE_INCLUDE = ["*attn*", "*mlp*"]
BASE_EXCLUDE = ["lm_head", "*router*"]
DEFAULT_RANDOM_SIGN_ROTATION = {
    **BASE_ROTATION,
    "rotation_kwargs": {"block_size": 16, "seed": 1, "random_sign": True},
}
MULTI_GEMM_ROTATION = {**BASE_ROTATION, "gemms": ["fwd", "wgrad"]}
BLOCK_32_ROTATION = {
    **BASE_ROTATION,
    "rotation_kwargs": {"block_size": 32, "seed": 1},
}
SEED_2_ROTATION = {
    **BASE_ROTATION,
    "rotation_kwargs": {"block_size": 16, "seed": 2},
}
UNSIGNED_ROTATION = {
    **BASE_ROTATION,
    "rotation_kwargs": {"block_size": 16, "seed": 1, "random_sign": False},
}
ROTATION_KEY_ROTATION_CASES = [
    (BASE_ROTATION, True),
    (DEFAULT_RANDOM_SIGN_ROTATION, True),
    (MULTI_GEMM_ROTATION, True),
    (BLOCK_32_ROTATION, False),
    (SEED_2_ROTATION, False),
    (UNSIGNED_ROTATION, False),
]
ROTATION_KEY_INCLUDE_CASES = [
    (BASE_INCLUDE, True),
    (BASE_INCLUDE[::-1], True),
    (["*mlp*"], False),
    ([], False),
]
ROTATION_KEY_EXCLUDE_CASES = [
    (BASE_EXCLUDE, True),
    (BASE_EXCLUDE[::-1], True),
    (["lm_head"], False),
]


@pytest.mark.parametrize("rotation_case", ROTATION_KEY_ROTATION_CASES)
@pytest.mark.parametrize("include_case", ROTATION_KEY_INCLUDE_CASES)
@pytest.mark.parametrize("exclude_case", ROTATION_KEY_EXCLUDE_CASES)
def test_build_rotation_key_identity(rotation_case, include_case, exclude_case):
    rotation, rotation_equal = rotation_case
    include, include_equal = include_case
    exclude, exclude_equal = exclude_case

    key = build_rotation_key(rotation, include, exclude)
    base_key = build_rotation_key(BASE_ROTATION, BASE_INCLUDE, BASE_EXCLUDE)

    assert key.startswith("hadamard-")
    assert "." not in key
    nn.ModuleDict({key: HadamardRotation(block_size=16)})
    assert (key == base_key) is (rotation_equal and include_equal and exclude_equal)


def test_build_rotation_key_reproducible_across_processes():
    """hashlib is unsalted, so the key cannot depend on process-local state."""
    script = (
        "from src.quant.rotation import build_rotation_key;"
        f"print(build_rotation_key({BASE_ROTATION!r}, {BASE_INCLUDE!r}, {BASE_EXCLUDE!r}))"
    )
    keys = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1")
    }

    assert keys == {build_rotation_key(BASE_ROTATION, BASE_INCLUDE, BASE_EXCLUDE)}
