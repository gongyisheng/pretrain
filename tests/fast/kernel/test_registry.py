import inspect
from collections.abc import Mapping
from typing import Any

import pytest

from src.kernel.ops import gemm as _gemm  # noqa: F401
from src.kernel.registry import KERNEL_REGISTRY, register_kernel
from src.kernel import spec as kernel_spec
from src.kernel.spec import CheckResult, KernelSpec


def _can_implement(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
    del args, kwargs
    return CheckResult(True)


def _identity(value: Any) -> Any:
    return value


def test_kernel_spec_exposes_one_implementation_check():
    spec = KernelSpec(
        op="test.contract",
        backend="eager",
        fn=_identity,
        can_implement=_can_implement,
        build="eager",
        autograd=True,
    )

    assert spec.can_implement is _can_implement
    assert not hasattr(spec, "availability")
    assert not hasattr(spec, "eligibility")


def test_register_kernel_accepts_only_can_implement_check():
    parameters = inspect.signature(register_kernel).parameters

    assert "can_implement" in parameters
    assert "priority" not in parameters
    assert "availability" not in parameters
    assert "eligibility" not in parameters


def test_register_kernel_records_can_implement_callback():
    @register_kernel(
        op="test.register_can_implement",
        backend="eager",
        can_implement=_can_implement,
        build="eager",
        autograd=True,
    )
    def registered(value: Any) -> Any:
        return value

    specs = KERNEL_REGISTRY.implementations("test.register_can_implement")

    assert len(specs) == 1
    assert specs[0].fn is registered
    assert specs[0].can_implement is _can_implement


@pytest.mark.parametrize(
    ("op", "expected_backends"),
    [
        ("gemm.grouped", {"eager", "torch", "triton"}),
        ("gemm.scaled", {"cublaslt", "eager", "torch", "triton"}),
        ("gemm.scaled_grouped", {"eager", "torch", "triton"}),
    ],
)
def test_registered_gemm_backends(op: str, expected_backends: set[str]):
    implementations = KERNEL_REGISTRY.implementations(op)
    by_backend = {spec.backend: spec for spec in implementations}

    assert set(by_backend) == expected_backends
    assert all(callable(spec.can_implement) for spec in implementations)


def test_backend_priorities_are_defined_once_per_backend():
    assert kernel_spec.BACKEND_PRIORITIES == {
        "eager": 0,
        "torch": 1,
        "triton": 2,
        "cublaslt": 3,
    }


def test_kernel_spec_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown kernel backend 'unknown'"):
        KernelSpec(
            op="test.unknown",
            backend="unknown",
            fn=_identity,
            can_implement=_can_implement,
            build="eager",
            autograd=True,
        )


def test_register_kernel_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown kernel backend 'unknown'"):

        @register_kernel(
            op="test.unknown",
            backend="unknown",
            can_implement=_can_implement,
            build="eager",
            autograd=True,
        )
        def unknown_backend(value: Any) -> Any:
            return value


def test_duplicate_registration_raises_value_error():
    registry = type(KERNEL_REGISTRY)()
    spec = KernelSpec(
        op="test.duplicate",
        backend="eager",
        fn=_identity,
        can_implement=_can_implement,
        build="eager",
        autograd=True,
    )
    registry.register(spec)

    with pytest.raises(ValueError, match="kernel already registered"):
        registry.register(spec)
