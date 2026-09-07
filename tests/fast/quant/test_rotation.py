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
GEMM_ATOL = {
    torch.float32: 4.6e-5,
    torch.float16: 0.065,
    torch.bfloat16: 0.52,
}
ROTATION_DEVICES = ("cpu", "cuda")
LARGE_ROTATION_ATOL = {
    # Worst dense-oracle errors are 1.18e-3 and 4.96e30; margins are 3.5x and 5.2x.
    torch.float16: 0.0041,
    torch.bfloat16: 2.6e31,
}
INVALID_HADAMARD_KWARGS = [
    {"random_sign": 1},
    {"seed": -1},
    {"seed": True},
    {"block_size": 4, "sign_vector": [1.0, -1.0, 0.0, 1.0]},
    {"unknown": 1},
]
SYLVESTER_4 = torch.tensor(
    [
        [0.5, 0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
    ]
)
SIGN_VECTOR_4 = torch.tensor([1.0, -1.0, -1.0, 1.0])
HADAMARD_SIGN_VECTORS = (None, SIGN_VECTOR_4)
HADAMARD_DIRECTIONS = (False, True)
HADAMARD_ORACLE_INPUT = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
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
    torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        transformed.square().sum(), x.square().sum(), atol=1e-4, rtol=1e-4
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


@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("sign_vector", HADAMARD_SIGN_VECTORS)
@pytest.mark.parametrize("inverse", HADAMARD_DIRECTIONS)
def test_hadamard_rotation_apply_oracle(contract_dim, sign_vector, inverse):
    rotation = HadamardRotation(
        block_size=4, random_sign=False, sign_vector=sign_vector
    )
    transform = SYLVESTER_4
    if sign_vector is not None:
        transform = (
            transform * sign_vector
            if inverse
            else sign_vector.unsqueeze(-1) * transform
        )
    x = HADAMARD_ORACLE_INPUT
    expected = (x.movedim(contract_dim, -1) @ transform).movedim(-1, contract_dim)
    transformed = (
        rotation.inverse(x, contract_dim) if inverse else rotation(x, contract_dim)
    )

    assert torch.equal(transformed, expected)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hadamard_rotation_apply_precision(dtype):
    block_size = 32
    x = torch.full(
        (3, block_size), torch.finfo(dtype).max / 8, dtype=dtype, device="cuda"
    )
    x[1, 1::2] *= -1
    x[2].zero_()
    x[2, : block_size // 2] = torch.finfo(dtype).max * 0.3
    signs = torch.tensor(
        [
            [
                1.0 if (row & col).bit_count() % 2 == 0 else -1.0
                for col in range(block_size)
            ]
            for row in range(block_size)
        ],
        device=x.device,
    )
    expected = (x.float() @ (signs / block_size**0.5)).to(dtype)

    transformed = HadamardRotation(block_size=block_size, random_sign=False)(x, -1)

    assert torch.isfinite(transformed).all()
    torch.testing.assert_close(
        transformed, expected, atol=LARGE_ROTATION_ATOL[dtype], rtol=0
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
ROTATION_KEY_IDENTITY_EQUAL_CASES = [
    {"rotation": DEFAULT_RANDOM_SIGN_ROTATION},
    {"rotation": MULTI_GEMM_ROTATION},
    {"include": BASE_INCLUDE[::-1]},
    {"exclude": BASE_EXCLUDE[::-1]},
]
ROTATION_KEY_IDENTITY_DIFFERENT_CASES = [
    {"rotation": BLOCK_32_ROTATION},
    {"rotation": SEED_2_ROTATION},
    {"rotation": UNSIGNED_ROTATION},
    {"include": ["*mlp*"]},
    {"include": []},
    {"exclude": ["lm_head"]},
]


def _key(rotation=None, include=None, exclude=None):
    return build_rotation_key(
        BASE_ROTATION if rotation is None else rotation,
        BASE_INCLUDE if include is None else include,
        BASE_EXCLUDE if exclude is None else exclude,
    )


@pytest.mark.parametrize("key_kwargs", ROTATION_KEY_IDENTITY_EQUAL_CASES)
def test_build_rotation_key_identity_equal(key_kwargs):
    key = _key(**key_kwargs)

    assert key.startswith("hadamard-")
    assert "." not in key
    nn.ModuleDict({key: HadamardRotation(block_size=16)})
    assert key == _key()


@pytest.mark.parametrize("key_kwargs", ROTATION_KEY_IDENTITY_DIFFERENT_CASES)
def test_build_rotation_key_identity_different(key_kwargs):
    assert _key(**key_kwargs) != _key()


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

    assert keys == {_key()}
