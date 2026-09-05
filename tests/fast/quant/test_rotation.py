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
ROTATION_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    ),
]
LARGE_ROTATION_ATOL = {
    # Worst dense-oracle errors are 1.18e-3 and 4.96e30; margins are 3.5x and 5.2x.
    torch.float16: 0.0041,
    torch.bfloat16: 2.6e31,
}
INVALID_HADAMARD_KWARGS = [
    {"block_size": 0},
    {"block_size": 3},
    {"block_size": 1.5},
    {"block_size": True},
    {"random_sign": 1},
    {"seed": -1},
    {"seed": True},
    {"block_size": 4, "sign_vector": [1.0, -1.0]},
    {"block_size": 4, "sign_vector": [[1.0], [-1.0], [-1.0], [1.0]]},
    {"block_size": 4, "sign_vector": [1.0, -1.0, 0.0, 1.0]},
    {"block_size": 4, "sign_vector": "bad"},
    {"unknown": 1},
]
INVALID_BUILD_ROTATION_KWARGS = [{"unknown": 1}, {"block_size": 3}]
SIGNED_HADAMARD_ORACLE_CASES = [
    pytest.param(
        -1,
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.tensor([[0.0, -2.0, -1.0, 5.0]]),
        torch.tensor([[0.0, -2.0, -1.0, 5.0]]),
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        id="contract-last-dimension",
    ),
    pytest.param(
        -2,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([[0.0, 0.0], [-4.0, -4.0], [-2.0, -2.0], [8.0, 10.0]]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([[8.0, 10.0], [2.0, 2.0], [4.0, 4.0], [0.0, 0.0]]),
        id="contract-penultimate-dimension",
    ),
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


@pytest.mark.parametrize("rotation_kwargs", INVALID_BUILD_ROTATION_KWARGS)
def test_build_rotation_raise_error(rotation_kwargs):
    with pytest.raises(ValueError, match="invalid quant rotation"):
        build_rotation(
            {
                "rotation_cls": "hadamard",
                "rotation_kwargs": rotation_kwargs,
            }
        )


def test_hadamard_rotation_apply_sylvester():
    rotation = HadamardRotation(block_size=4, random_sign=False)
    basis = torch.eye(4)
    expected = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
        ]
    )

    transformed = rotation(basis, -1)

    assert torch.equal(transformed, expected)


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


@pytest.mark.parametrize(
    "contract_dim, x, expected_transformed, inverse_input, expected_inverse",
    SIGNED_HADAMARD_ORACLE_CASES,
)
def test_hadamard_rotation_signed_oracle(
    contract_dim, x, expected_transformed, inverse_input, expected_inverse
):
    signs = torch.tensor([1.0, -1.0, -1.0, 1.0])
    rotation = HadamardRotation(block_size=4, sign_vector=signs)

    assert torch.equal(rotation(x, contract_dim), expected_transformed)
    assert torch.equal(rotation.inverse(inverse_input, contract_dim), expected_inverse)


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hadamard_rotation_apply_dtype_device(dtype, device):
    torch.manual_seed(0)
    rotation = HadamardRotation(block_size=8, random_sign=False)
    x = torch.randn(16, 64, dtype=dtype, device=device)
    other_device = "cuda" if device == "cpu" else "cpu"

    with torch.device(other_device):
        transformed = rotation(x, -1)

    assert transformed.dtype == dtype
    assert transformed.device == x.device


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


@pytest.mark.parametrize("device", ROTATION_DEVICES)
@pytest.mark.parametrize("contract_dim", CONTRACT_DIMS)
@pytest.mark.parametrize("dtype", GEMM_DTYPES)
def test_hadamard_rotation_apply_out_dtype(dtype, contract_dim, device):
    """Promoting on the store is exactly the pre-cast it replaces, in one pass."""
    torch.manual_seed(0)
    rotation = HadamardRotation(block_size=8, seed=7).to(device)
    x = torch.randn(64, 128, dtype=dtype, device=device)

    promoted = rotation(x, contract_dim, torch.float32)

    assert promoted.dtype is torch.float32
    assert torch.equal(promoted, rotation(x.float(), contract_dim))
    assert torch.equal(
        rotation.inverse(x, contract_dim, torch.float32),
        rotation.inverse(x.float(), contract_dim),
    )


def test_hadamard_rotation_apply_block_one_identity():
    rotation = HadamardRotation(block_size=1)
    x = torch.randn(3, 4)

    assert rotation(x, -1) is x
    assert rotation.inverse(x, -2) is x
    assert rotation(x, -1, torch.bfloat16).dtype is torch.bfloat16


def test_hadamard_rotation_apply_raise_error():
    rotation = HadamardRotation(block_size=8, random_sign=False)

    with pytest.raises(ValueError, match="block 8 must divide"):
        rotation(torch.randn(8, 10), -1)
    with pytest.raises(ValueError, match="contract_dim must be -2 or -1"):
        rotation(torch.randn(8, 16), 0)


# --- rotation identity -------------------------------------------------------

BASE_ROTATION = {
    "rotation_cls": "hadamard",
    "rotation_kwargs": {"block_size": 16, "seed": 1},
    "gemms": ["fwd"],
}
BASE_INCLUDE = ["*attn*", "*mlp*"]
BASE_EXCLUDE = ["lm_head", "*router*"]
ROTATION_KEY_IDENTITY_CASES = [
    pytest.param(
        {
            **BASE_ROTATION,
            "rotation_kwargs": {"block_size": 16, "seed": 1, "random_sign": True},
        },
        None,
        None,
        True,
        id="same-transform-default-random-sign",
    ),
    pytest.param(
        {**BASE_ROTATION, "gemms": ["fwd", "wgrad"]},
        None,
        None,
        True,
        id="same-transform-different-gemms",
    ),
    pytest.param(
        {**BASE_ROTATION, "rotation_kwargs": {"block_size": 32, "seed": 1}},
        None,
        None,
        False,
        id="different-block-size",
    ),
    pytest.param(
        {**BASE_ROTATION, "rotation_kwargs": {"block_size": 16, "seed": 2}},
        None,
        None,
        False,
        id="different-seed",
    ),
    pytest.param(
        {
            **BASE_ROTATION,
            "rotation_kwargs": {"block_size": 16, "seed": 1, "random_sign": False},
        },
        None,
        None,
        False,
        id="different-random-sign",
    ),
    pytest.param(
        BASE_ROTATION,
        ["*mlp*"],
        None,
        False,
        id="different-include-scope",
    ),
    pytest.param(
        BASE_ROTATION,
        None,
        ["lm_head"],
        False,
        id="different-exclude-scope",
    ),
    pytest.param(
        BASE_ROTATION,
        [],
        None,
        False,
        id="empty-include-scope",
    ),
    pytest.param(
        BASE_ROTATION,
        BASE_INCLUDE[::-1],
        BASE_EXCLUDE[::-1],
        True,
        id="glob-order-does-not-matter",
    ),
]


def _key(rotation=None, include=None, exclude=None):
    return build_rotation_key(
        BASE_ROTATION if rotation is None else rotation,
        BASE_INCLUDE if include is None else include,
        BASE_EXCLUDE if exclude is None else exclude,
    )


def test_build_rotation_key_is_a_legal_state_dict_path():
    key = _key()

    assert key.startswith("hadamard-")
    assert "." not in key
    nn.ModuleDict({key: HadamardRotation(block_size=16)})


@pytest.mark.parametrize(
    "rotation, include, exclude, expected_equal",
    ROTATION_KEY_IDENTITY_CASES,
)
def test_build_rotation_key_identity(rotation, include, exclude, expected_equal):
    assert (_key(rotation, include, exclude) == _key()) is expected_equal


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
