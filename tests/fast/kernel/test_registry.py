import inspect
from collections.abc import Mapping
from typing import Any

import pytest

from src.kernel import spec as kernel_spec
from src.kernel.ops import gemm as _gemm  # noqa: F401
from src.kernel.registry import (
    KERNEL_REGISTRY,
    KernelRegistry,
    register_kernel,
    register_operation,
)
from src.kernel.spec import CheckResult, KernelSpec


def _can_implement(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
    del args, kwargs
    return CheckResult(True)


def _identity(value: Any) -> Any:
    return value


def _spec(op: str, backend: str = "eager") -> KernelSpec:
    return KernelSpec(
        op=op,
        backend=backend,
        fn=_identity,
        build="eager",
        autograd=True,
    )


def test_kernel_spec_contains_only_implementation_metadata():
    spec = _spec("test.contract")

    assert spec.fn is _identity
    assert not hasattr(spec, "can_implement")
    assert not hasattr(spec, "availability")
    assert not hasattr(spec, "eligibility")


def test_register_kernel_does_not_accept_implementation_check():
    parameters = inspect.signature(register_kernel).parameters

    assert "can_implement" not in parameters
    assert "priority" not in parameters
    assert "availability" not in parameters
    assert "eligibility" not in parameters


def test_register_kernel_records_implementation_metadata():
    op = "test.register_metadata"
    register_operation(op, {"eager": _can_implement})

    @register_kernel(op=op, backend="eager", build="eager", autograd=True)
    def registered(value: Any) -> Any:
        return value

    specs = KERNEL_REGISTRY.implementations(op)

    assert len(specs) == 1
    assert specs[0].fn is registered
    assert not hasattr(specs[0], "can_implement")


@pytest.mark.parametrize(
    ("op", "expected_backends", "can_implement"),
    [
        (
            "gemm.grouped",
            {"eager", "torch", "triton"},
            {
                "eager": _gemm.can_implement_grouped_gemm_eager,
                "torch": _gemm.can_implement_grouped_gemm_torch,
                "triton": _gemm.can_implement_grouped_gemm_triton,
            },
        ),
        (
            "gemm.scaled",
            {"cublaslt", "eager", "torch", "triton"},
            {
                "eager": _gemm.can_implement_scaled_gemm_eager,
                "torch": _gemm.can_implement_scaled_gemm_torch,
                "triton": _gemm.can_implement_scaled_gemm_triton,
                "cublaslt": _gemm.can_implement_scaled_gemm_cublaslt,
            },
        ),
        (
            "gemm.scaled_grouped",
            {"eager", "torch", "triton"},
            {
                "eager": _gemm.can_implement_scaled_grouped_gemm_eager,
                "torch": _gemm.can_implement_scaled_grouped_gemm_torch,
                "triton": _gemm.can_implement_scaled_grouped_gemm_triton,
            },
        ),
    ],
)
def test_registered_gemm_backends(
    op: str,
    expected_backends: set[str],
    can_implement: Mapping[str, Any],
):
    implementations = KERNEL_REGISTRY.implementations(op)
    operation = KERNEL_REGISTRY.operation(op)

    assert operation is not None
    assert {spec.backend for spec in implementations} == expected_backends
    assert set(operation.can_implement) == expected_backends
    for backend, validator in can_implement.items():
        assert KERNEL_REGISTRY.can_implement(op, backend) is validator


def test_backend_priorities_are_defined_once_per_backend():
    assert kernel_spec.BACKEND_PRIORITIES == {
        "eager": 0,
        "torch": 1,
        "triton": 2,
        "cublaslt": 3,
    }


def test_kernel_spec_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown kernel backend 'unknown'"):
        _spec("test.unknown", "unknown")


def test_register_kernel_rejects_unknown_backend():
    op = "test.unknown"
    register_operation(op, {"eager": _can_implement})

    with pytest.raises(ValueError, match="unknown kernel backend 'unknown'"):

        @register_kernel(op=op, backend="unknown", build="eager", autograd=True)
        def unknown_backend(value: Any) -> Any:
            return value


def test_duplicate_registration_raises_value_error():
    registry = KernelRegistry()
    spec = _spec("test.duplicate")
    registry.register_operation(spec.op, {"eager": _can_implement})
    registry.register(spec)

    with pytest.raises(ValueError, match="kernel already registered"):
        registry.register(spec)


def test_register_operation_records_backend_checks():
    registry = KernelRegistry()

    registry.register_operation("test.operation", {"eager": _no})

    assert registry.can_implement("test.operation", "eager") is _no


def _no(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
    del args, kwargs
    return CheckResult(False, "unsupported")


def test_register_operation_copies_backend_mapping():
    registry = KernelRegistry()
    callbacks = {"eager": _can_implement}
    registry.register_operation("test.immutable", callbacks)
    callbacks["torch"] = _can_implement
    operation = registry.operation("test.immutable")

    assert operation is not None
    with pytest.raises(TypeError):
        operation.can_implement["torch"] = _can_implement

    with pytest.raises(ValueError, match="validator"):
        registry.register(_spec("test.immutable", "torch"))


def test_register_operation_rejects_duplicate_metadata():
    registry = KernelRegistry()
    registry.register_operation("test.operation", {"eager": _can_implement})

    with pytest.raises(ValueError, match="operation already registered"):
        registry.register_operation("test.operation", {"eager": _can_implement})


def test_register_operation_rejects_empty_and_unknown_backend_mappings():
    registry = KernelRegistry()

    with pytest.raises(ValueError, match="nonempty"):
        registry.register_operation("test.empty", {})
    assert registry.operation("test.empty") is None
    with pytest.raises(ValueError, match="unknown kernel backend 'unknown'"):
        registry.register_operation("test.unknown", {"unknown": _can_implement})
    assert registry.operation("test.unknown") is None


@pytest.mark.parametrize(
    ("op", "backends"),
    [
        ("test.invalid_backend", {"eager": None}),
    ],
)
def test_register_operation_rejects_non_callable_checks(op, backends):
    registry = KernelRegistry()

    with pytest.raises(TypeError, match="callable"):
        registry.register_operation(op, backends)

    assert registry.operation(op) is None


def test_operation_first_registration_requires_a_mapped_backend_validator():
    registry = KernelRegistry()
    registry.register_operation("test.operation_first", {"torch": _can_implement})

    with pytest.raises(ValueError, match="validator"):
        registry.register(_spec("test.operation_first", "eager"))

    assert registry.implementations("test.operation_first") == ()


def test_kernel_first_registration_requires_a_mapped_backend_validator():
    registry = KernelRegistry()
    registry.register(_spec("test.kernel_first", "eager"))

    with pytest.raises(ValueError, match="validator"):
        registry.register_operation("test.kernel_first", {"torch": _can_implement})

    assert registry.operation("test.kernel_first") is None
    assert {spec.backend for spec in registry.implementations("test.kernel_first")} == {
        "eager"
    }


def test_kernel_first_registration_accepts_a_complete_backend_mapping():
    registry = KernelRegistry()
    registry.register(_spec("test.kernel_first_complete", "eager"))

    registry.register_operation("test.kernel_first_complete", {"eager": _can_implement})

    assert (
        registry.can_implement("test.kernel_first_complete", "eager") is _can_implement
    )


def test_global_register_operation_records_backend_checks():
    op = "test.global_operation"
    register_operation(op, {"eager": _can_implement})

    assert KERNEL_REGISTRY.can_implement(op, "eager") is _can_implement
