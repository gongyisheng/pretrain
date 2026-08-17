import ast
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.kernel.ops import gemm as gemm_module
from src.kernel.spec import CheckResult


class _TensorMeta(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: str = "cuda:7",
        contiguous: bool = True,
        strides: tuple[int, ...] | None = None,
    ) -> "_TensorMeta":
        instance = torch.Tensor._make_subclass(
            cls,
            torch.empty(shape, dtype=dtype, device="meta"),
            require_grad=False,
        )
        instance._device = torch.device(device)
        instance._contiguous = contiguous
        if strides is None:
            running = 1
            calculated = []
            for size in reversed(shape):
                calculated.append(running)
                running *= size
            instance._strides = tuple(reversed(calculated))
        else:
            instance._strides = strides
        return instance

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self, dim: int | None = None):
        return self._strides if dim is None else self._strides[dim]

    def element_size(self) -> int:
        return self.dtype.itemsize

    def numel(self) -> int:
        total = 1
        for size in self.shape:
            total *= size
        return total


_REPOSITORY_ROOT = Path(__file__).parents[4]
_BACKEND_MODULES = (
    "src/kernel/backends/eager/__init__.py",
    "src/kernel/backends/eager/gemm.py",
    "src/kernel/backends/triton/__init__.py",
    "src/kernel/backends/triton/gemm.py",
    "src/kernel/backends/cublaslt/__init__.py",
    "src/kernel/backends/cublaslt/gemm.py",
)


@pytest.fixture
def available_cublaslt_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemm_module, "_CUBLASLT_EXTENSION_AVAILABLE", True)
    monkeypatch.setattr(gemm_module, "_CUBLASLT_EXTENSION_ERROR", None)


@pytest.mark.parametrize("relative_path", _BACKEND_MODULES)
def test_backend_modules_do_not_import_ops(relative_path: str):
    tree = ast.parse((_REPOSITORY_ROOT / relative_path).read_text())
    imported_modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    imported_modules.extend(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(
        module == "src.kernel.ops" or module.startswith("src.kernel.ops.")
        for module in imported_modules
    )


def test_direct_backend_import_does_not_initialize_operation_metadata():
    script = """
import src.kernel.backends.eager.gemm
from src.kernel.selector import KernelSelectionError, select_kernel

try:
    select_kernel("gemm.grouped", (), {})
except KernelSelectionError as error:
    assert "src.kernel.ops" in str(error)
else:
    raise AssertionError("selection unexpectedly succeeded without operation metadata")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_importing_gemm_ops_preloads_cublaslt_extension_status():
    script = """
from src.kernel.backends import cublaslt as cublaslt_backend
from src.kernel.ops import gemm

assert gemm._CUBLASLT_EXTENSION_AVAILABLE == (cublaslt_backend._C is not None)
assert gemm._CUBLASLT_EXTENSION_ERROR == cublaslt_backend._C_IMPORT_ERROR
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _grouped_metadata(layout: str, with_bias: bool):
    offs = _TensorMeta((2,), torch.int32)
    bias = _TensorMeta((2, 48), torch.bfloat16) if with_bias else None
    if layout == "ragged_m":
        return (
            _TensorMeta((64, 64), torch.bfloat16),
            _TensorMeta((2, 64, 48), torch.bfloat16),
            offs,
            bias,
        )
    if layout == "ragged_k":
        return (
            _TensorMeta((32, 64), torch.bfloat16),
            _TensorMeta((64, 48), torch.bfloat16),
            offs,
            bias,
        )
    return (
        _TensorMeta((2, 32, 64), torch.bfloat16),
        _TensorMeta((64, 48), torch.bfloat16),
        offs,
        bias,
    )


def _default_grouped_metadata():
    return _grouped_metadata("ragged_m", False)


def test_eager_grouped_requires_matching_allowed_operand_dtypes():
    args = (
        torch.empty(64, 64, dtype=torch.float16),
        torch.empty(2, 64, 48, dtype=torch.float32),
        torch.tensor([32, 64], dtype=torch.int32),
        None,
    )

    result = gemm_module.can_implement_grouped_gemm_eager(args, {})

    assert not result.ok
    assert result.reason is not None
    assert "same dtype" in result.reason


@pytest.mark.parametrize(
    ("predicate", "expected_argument_count"),
    [
        (gemm_module.can_implement_grouped_gemm_eager, 4),
        (gemm_module.can_implement_scaled_gemm_eager, 8),
        (gemm_module.can_implement_scaled_grouped_gemm_eager, 9),
    ],
)
def test_eager_gemm_predicates_reject_wrong_argument_count(
    predicate, expected_argument_count: int
):
    result = predicate((), {})

    assert result == CheckResult(False, f"expected {expected_argument_count} arguments")


@pytest.mark.parametrize(
    ("predicate", "expected_argument_count"),
    [
        (gemm_module.can_implement_grouped_gemm_eager, 4),
        (gemm_module.can_implement_grouped_gemm_triton, 4),
        (gemm_module.can_implement_scaled_gemm_eager, 8),
        (gemm_module.can_implement_scaled_gemm_triton, 8),
        (gemm_module.can_implement_scaled_gemm_cublaslt, 8),
        (gemm_module.can_implement_scaled_grouped_gemm_eager, 9),
        (gemm_module.can_implement_scaled_grouped_gemm_triton, 9),
        (gemm_module.can_implement_scaled_grouped_gemm_cublaslt, 9),
    ],
)
def test_all_gemm_backend_validators_reject_malformed_common_contracts(
    predicate, expected_argument_count: int
):
    assert predicate((), {}) == CheckResult(
        False, f"expected {expected_argument_count} arguments"
    )


@pytest.mark.parametrize(
    "predicate",
    [
        gemm_module.can_implement_grouped_gemm_eager,
    ],
)
def test_grouped_backends_reject_empty_offsets(
    monkeypatch: pytest.MonkeyPatch, predicate
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_default_grouped_metadata())
    args[2] = _TensorMeta((0,), torch.int32)

    result = predicate(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "nonempty" in result.reason


def test_eager_grouped_rejects_empty_offsets():
    args = (
        torch.empty(64, 64, dtype=torch.bfloat16),
        torch.empty(2, 64, 48, dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.int32),
        None,
    )

    result = gemm_module.can_implement_grouped_gemm_eager(args, {})

    assert not result.ok
    assert result.reason is not None
    assert "nonempty" in result.reason


def test_eager_grouped_rejects_ragged_n_bias():
    result = gemm_module.can_implement_grouped_gemm_eager(
        _grouped_metadata(layout="ragged_n", with_bias=True), {}
    )

    assert not result.ok
    assert result.reason is not None
    assert "ragged-N" in result.reason


def _scaled_metadata(device: str = "cuda:7"):
    return (
        _TensorMeta((64, 64), torch.float8_e4m3fn, device),
        _TensorMeta((64, 48), torch.float8_e4m3fn, device),
        _TensorMeta((64, 1), torch.float32, device),
        _TensorMeta((1, 48), torch.float32, device),
        torch.bfloat16,
        0,
        torch.float32,
        None,
    )


def test_eager_scaled_checks_bias_after_common_validation():
    args = list(_scaled_metadata())
    args[4] = torch.float64
    args[7] = _TensorMeta((48,), torch.int8)

    result = gemm_module.can_implement_scaled_gemm_eager(tuple(args), {})

    assert result.reason is not None
    assert "output dtype" in result.reason


def _cublaslt_scaled_metadata():
    return (
        _TensorMeta((129, 144), torch.float8_e4m3fn),
        _TensorMeta((144, 160), torch.float8_e4m3fn),
        _TensorMeta((129, 5), torch.float32),
        _TensorMeta((5, 160), torch.float32),
        torch.bfloat16,
        32,
        torch.float8_e8m0fnu,
        None,
    )


def _scaled_grouped_metadata(device: str = "cuda:7"):
    return (
        _TensorMeta((64, 64), torch.float8_e4m3fn, device),
        _TensorMeta((2, 64, 48), torch.float8_e4m3fn, device),
        _TensorMeta((64, 1), torch.float32, device),
        _TensorMeta((2, 1, 48), torch.float32, device),
        _TensorMeta((2,), torch.int32, device),
        torch.bfloat16,
        0,
        torch.float32,
        None,
    )


def _cublaslt_scaled_grouped_metadata(device: str = "cuda:7"):
    return (
        _TensorMeta((64, 64), torch.float8_e4m3fn, device),
        _TensorMeta((2, 64, 48), torch.float8_e4m3fn, device),
        _TensorMeta((64, 2), torch.float32, device),
        _TensorMeta((2, 2, 48), torch.float32, device),
        _TensorMeta((2,), torch.int32, device),
        torch.bfloat16,
        32,
        torch.float8_e8m0fnu,
        None,
    )


def test_cublaslt_scaled_grouped_gemm_accepts_ragged_m_sm100(
    monkeypatch: pytest.MonkeyPatch,
    available_cublaslt_extension: None,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))

    result = gemm_module.can_implement_scaled_grouped_gemm_cublaslt(
        _cublaslt_scaled_grouped_metadata(), {}
    )

    assert result.ok, result.reason


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda args: (
                args.__setitem__(1, _TensorMeta((64, 48), torch.float8_e4m3fn)),
                args.__setitem__(3, _TensorMeta((2, 48), torch.float32)),
            ),
            "requires rank in {3}",
        ),
        (
            lambda args: args.__setitem__(4, _TensorMeta((2,), torch.int64)),
            "offsets received dtype",
        ),
        (
            lambda args: (
                args.__setitem__(2, _TensorMeta((64, 1), torch.float32)),
                args.__setitem__(3, _TensorMeta((2, 1, 48), torch.float32)),
                args.__setitem__(6, 64),
            ),
            "block_size",
        ),
        (lambda args: args.__setitem__(5, torch.float16), "output dtype"),
        (
            lambda args: args.__setitem__(8, _TensorMeta((2, 48), torch.bfloat16)),
            "does not support bias",
        ),
        (
            lambda args: args.__setitem__(3, _TensorMeta((2, 1, 48), torch.float32)),
            "B scale",
        ),
    ],
)
def test_cublaslt_scaled_grouped_gemm_rejects_unsupported_contracts(
    monkeypatch: pytest.MonkeyPatch,
    available_cublaslt_extension: None,
    mutation,
    reason: str,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    args = list(_cublaslt_scaled_grouped_metadata())
    mutation(args)

    result = gemm_module.can_implement_scaled_grouped_gemm_cublaslt(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert reason in result.reason


def test_cublaslt_scaled_grouped_gemm_rejects_unverified_sm120(
    monkeypatch: pytest.MonkeyPatch,
    available_cublaslt_extension: None,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = gemm_module.can_implement_scaled_grouped_gemm_cublaslt(
        _cublaslt_scaled_grouped_metadata(), {}
    )

    assert not result.ok
    assert result.reason is not None
    assert "SM10x or SM110" in result.reason


@pytest.mark.parametrize(
    "predicate",
    [
        gemm_module.can_implement_scaled_grouped_gemm_eager,
    ],
)
def test_scaled_grouped_backends_reject_empty_offsets(
    monkeypatch: pytest.MonkeyPatch, predicate
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_scaled_grouped_metadata())
    args[4] = _TensorMeta((0,), torch.int32)

    result = predicate(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "nonempty" in result.reason


def test_eager_scaled_grouped_rejects_empty_offsets():
    args = (
        torch.empty(64, 64, dtype=torch.float8_e4m3fn),
        torch.empty(2, 64, 48, dtype=torch.float8_e4m3fn),
        torch.empty(64, 1, dtype=torch.float32),
        torch.empty(2, 1, 48, dtype=torch.float32),
        torch.empty(0, dtype=torch.int32),
        torch.bfloat16,
        0,
        torch.float32,
        None,
    )

    result = gemm_module.can_implement_scaled_grouped_gemm_eager(args, {})

    assert not result.ok
    assert result.reason is not None
    assert "nonempty" in result.reason


def _triton_scaled_metadata(block_size: int = 0, k: int = 64):
    nblocks = (k + block_size - 1) // block_size if block_size else 1
    return (
        _TensorMeta((64, k), torch.float8_e4m3fn),
        _TensorMeta((k, 48), torch.float8_e4m3fn),
        _TensorMeta((64, nblocks), torch.float32),
        _TensorMeta((nblocks, 48), torch.float32),
        torch.float32,
        block_size,
        torch.float32,
        None,
    )


@pytest.mark.parametrize(
    ("predicate_name", "args_factory"),
    [
        ("can_implement_grouped_gemm_eager", _default_grouped_metadata),
        ("can_implement_scaled_gemm_eager", _scaled_metadata),
        ("can_implement_scaled_grouped_gemm_eager", _scaled_grouped_metadata),
    ],
)
def test_shared_gemm_capability_predicates_accept_valid_contracts(
    predicate_name: str, args_factory
):
    predicate = getattr(gemm_module, predicate_name)
    result = predicate(args_factory(), {})

    assert result.ok, result.reason


@pytest.mark.parametrize(
    ("predicate", "args_factory", "scale_dtype_index"),
    [
        (gemm_module.can_implement_scaled_gemm_eager, _scaled_metadata, 6),
        (
            gemm_module.can_implement_scaled_grouped_gemm_eager,
            _scaled_grouped_metadata,
            7,
        ),
    ],
)
@pytest.mark.parametrize("invalid_scale_dtype", [None, "invalid"])
def test_shared_scaled_gemm_predicates_reject_invalid_scale_dtype(
    predicate, args_factory, scale_dtype_index: int, invalid_scale_dtype
):
    args = list(args_factory())
    args[scale_dtype_index] = invalid_scale_dtype

    result = predicate(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "scale_dtype must be torch.dtype" in result.reason


@pytest.mark.parametrize(
    ("name", "mutate", "expected_reason"),
    [
        (
            "e5m2_operands",
            lambda args: args.__setitem__(
                0, _TensorMeta((129, 144), torch.float8_e5m2)
            ),
            "operands",
        ),
        (
            "wrong_scale_dtype",
            lambda args: args.__setitem__(6, torch.float32),
            "scale_dtype",
        ),
        ("fp16_output", lambda args: args.__setitem__(4, torch.float16), "bfloat16"),
        (
            "invalid_bias",
            lambda args: args.__setitem__(7, _TensorMeta((160,), torch.float16)),
            "dtype",
        ),
    ],
)
def test_cublaslt_scaled_gemm_rejects_invalid_contracts(
    monkeypatch: pytest.MonkeyPatch,
    available_cublaslt_extension: None,
    name: str,
    mutate,
    expected_reason: str,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_cublaslt_scaled_metadata())
    mutate(args)

    result = gemm_module.can_implement_scaled_gemm_cublaslt(tuple(args), {})

    assert not result.ok, name
    assert result.reason is not None
    assert expected_reason in result.reason


def test_cublaslt_scaled_gemm_accepts_sm120_contract(
    monkeypatch: pytest.MonkeyPatch,
    available_cublaslt_extension: None,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = gemm_module.can_implement_scaled_gemm_cublaslt(
        _cublaslt_scaled_metadata(), {}
    )

    assert result.ok


def test_cublaslt_scaled_gemm_requires_block_size_32(
    available_cublaslt_extension: None,
):
    args = list(_cublaslt_scaled_metadata())
    args[2] = _TensorMeta((129, 3), torch.float32)
    args[3] = _TensorMeta((3, 160), torch.float32)
    args[5] = 64

    result = gemm_module.can_implement_scaled_gemm_cublaslt(tuple(args), {})

    assert result == CheckResult(
        False,
        "MXFP8 GEMM block_size received value(s) 64; requires one of 32",
    )


def test_shared_scaled_gemm_rejects_invalid_ndim_before_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
):
    args = list(_cublaslt_scaled_metadata())
    args[0] = _TensorMeta((144,), torch.float8_e4m3fn)

    result = gemm_module.can_implement_scaled_gemm_eager(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "rank" in result.reason


def test_cublaslt_scaled_gemm_rejects_missing_extension(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(gemm_module, "_CUBLASLT_EXTENSION_AVAILABLE", False)
    monkeypatch.setattr(gemm_module, "_CUBLASLT_EXTENSION_ERROR", "not installed")

    result = gemm_module.can_implement_scaled_gemm_cublaslt(
        _cublaslt_scaled_metadata(), {}
    )

    assert not result.ok
    assert result.reason is not None
    assert result.reason == "cuBLASLt extension unavailable: not installed"


def test_cublaslt_scaled_gemm_checks_contract_before_sm100(
    monkeypatch: pytest.MonkeyPatch,
    available_cublaslt_extension: None,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    args = list(_cublaslt_scaled_metadata())
    args[0] = _TensorMeta((129, 144), torch.int8)

    result = gemm_module.can_implement_scaled_gemm_cublaslt(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "operands" in result.reason


@pytest.mark.parametrize(
    ("predicate", "args_factory", "index", "replacement"),
    [
        (
            gemm_module.can_implement_scaled_gemm_cublaslt,
            _cublaslt_scaled_metadata,
            6,
            torch.float32,
        ),
        (
            gemm_module.can_implement_grouped_gemm_triton,
            _default_grouped_metadata,
            2,
            _TensorMeta((2,), torch.int64),
        ),
    ],
)
def test_gemm_validation_checks_compute_capability_last(
    monkeypatch: pytest.MonkeyPatch,
    available_cublaslt_extension: None,
    predicate,
    args_factory,
    index: int,
    replacement,
):
    def unexpected_query(device: torch.device) -> tuple[int, int]:
        raise AssertionError(f"unexpected CUDA query for {device}")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_query)
    args = list(args_factory())
    args[index] = replacement

    result = predicate(tuple(args), {})

    assert not result.ok


@pytest.mark.parametrize(
    ("predicate", "args_factory", "index", "replacement", "reason"),
    [
        (
            gemm_module.can_implement_grouped_gemm_eager,
            _default_grouped_metadata,
            1,
            _TensorMeta((2, 32, 48), torch.bfloat16),
            "contraction",
        ),
        (
            gemm_module.can_implement_scaled_gemm_eager,
            _scaled_metadata,
            1,
            _TensorMeta((32, 48), torch.float8_e4m3fn),
            "contraction",
        ),
        (
            gemm_module.can_implement_scaled_gemm_eager,
            _scaled_metadata,
            2,
            _TensorMeta((64, 2), torch.float32),
            "A scale",
        ),
        (
            gemm_module.can_implement_scaled_gemm_eager,
            _scaled_metadata,
            5,
            32,
            "A scale",
        ),
        (
            gemm_module.can_implement_grouped_gemm_eager,
            _default_grouped_metadata,
            3,
            _TensorMeta((2, 47), torch.bfloat16),
            "shape",
        ),
    ],
)
def test_gemm_validation_rejects_shape_block_and_bias_contracts(
    monkeypatch: pytest.MonkeyPatch,
    predicate,
    args_factory,
    index: int,
    replacement,
    reason: str,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(args_factory())
    args[index] = replacement

    result = predicate(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert reason in result.reason


def test_triton_scaled_gemm_validation_requires_same_device():
    args = list(_scaled_metadata())
    args[2] = _TensorMeta((64, 1), torch.float32, "cuda:6")

    result = gemm_module.can_implement_scaled_gemm_eager(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "same device" in result.reason


def test_triton_scaled_gemm_rejects_mixed_operand_dtype_families():
    args = list(_triton_scaled_metadata())
    args[1] = _TensorMeta((64, 48), torch.int8)

    result = gemm_module.can_implement_scaled_gemm_triton(tuple(args), {})

    assert result == CheckResult(
        False,
        "scaled GEMM requires operands from the same dtype family",
    )


def test_triton_scaled_grouped_gemm_validation_rejects_malformed_scale_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_scaled_grouped_metadata())
    args[0] = _TensorMeta((4, 8), torch.int8)
    args[1] = _TensorMeta((2, 8, 6), torch.int8)
    args[2] = _TensorMeta((4, 2), torch.float32)
    args[3] = _TensorMeta((2, 2, 5), torch.float32)

    result = gemm_module.can_implement_scaled_grouped_gemm_triton(tuple(args), {})

    assert not result.ok
    assert result.reason == "scaled grouped GEMM has invalid A scale layout"


def test_triton_grouped_gemm_validation_rejects_bias_for_ragged_n(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_default_grouped_metadata())
    args[0] = _TensorMeta((2, 64, 64), torch.bfloat16)
    args[1] = _TensorMeta((64, 48), torch.bfloat16)
    args[3] = _TensorMeta((2, 48), torch.bfloat16)

    result = gemm_module.can_implement_grouped_gemm_eager(tuple(args), {})

    assert not result.ok
    assert result.reason == "bias not supported for ragged-N"


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("contraction", "contraction dimensions"),
        ("sa_rows", "A scale"),
        ("sa_1d", "rank"),
        ("sb_cols", "B scale"),
        ("sb_blocks", "B scale"),
        ("sb_1d", "rank"),
        ("scale_blocks_vs_block_size", "A scale"),
        ("bias_len", "bias"),
        ("bias_2d", "bias"),
        ("cross_family_dtype", "same dtype family"),
    ],
)
def test_triton_scaled_gemm_validation_rejects_malformed_metadata(
    monkeypatch: pytest.MonkeyPatch, case: str, reason: str
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_triton_scaled_metadata(block_size=32, k=256))
    if case == "contraction":
        args[1] = _TensorMeta((128, 48), torch.float8_e4m3fn)
    elif case == "sa_rows":
        args[2] = _TensorMeta((32, 8), torch.float32)
    elif case == "sa_1d":
        args[2] = _TensorMeta((64,), torch.float32)
    elif case == "sb_cols":
        args[3] = _TensorMeta((8, 24), torch.float32)
    elif case == "sb_blocks":
        args[3] = _TensorMeta((1, 48), torch.float32)
    elif case == "sb_1d":
        args[3] = _TensorMeta((48,), torch.float32)
    elif case == "scale_blocks_vs_block_size":
        args[2] = _TensorMeta((64, 1), torch.float32)
        args[3] = _TensorMeta((1, 48), torch.float32)
    elif case == "bias_len":
        args[7] = _TensorMeta((24,), torch.bfloat16)
    elif case == "bias_2d":
        args[7] = _TensorMeta((1, 48), torch.bfloat16)
    else:
        args[1] = _TensorMeta((256, 48), torch.int8)

    predicate = (
        gemm_module.can_implement_scaled_gemm_triton
        if case == "cross_family_dtype"
        else gemm_module.can_implement_scaled_gemm_eager
    )
    result = predicate(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert reason in result.reason


@pytest.mark.parametrize(("block_size", "nblocks"), [(8, 32), (96, 3)])
def test_triton_scaled_gemm_validation_rejects_unusable_block_size(
    monkeypatch: pytest.MonkeyPatch, block_size: int, nblocks: int
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_triton_scaled_metadata(block_size=block_size, k=256))

    assert args[2].shape == (64, nblocks)
    result = gemm_module.can_implement_scaled_gemm_triton(tuple(args), {})

    assert not result.ok
    assert result.reason == "block_size must be a power of two >= 16"


def test_triton_scaled_grouped_gemm_validation_rejects_bias_for_ragged_n(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_scaled_grouped_metadata())
    args[0] = _TensorMeta((2, 64, 64), torch.float8_e4m3fn)
    args[1] = _TensorMeta((64, 48), torch.float8_e4m3fn)
    args[2] = _TensorMeta((2, 64, 1), torch.float32)
    args[3] = _TensorMeta((1, 48), torch.float32)
    args[8] = _TensorMeta((2, 48), torch.bfloat16)

    result = gemm_module.can_implement_scaled_grouped_gemm_eager(tuple(args), {})

    assert not result.ok
    assert result.reason == "bias not supported for ragged-N"
