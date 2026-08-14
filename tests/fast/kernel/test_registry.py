import inspect
from collections.abc import Mapping
from typing import Any

from src.kernel.registry import KERNEL_REGISTRY, register_kernel
from src.kernel.spec import CheckResult, KernelSpec


def _can_implement(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
    del args, kwargs
    return CheckResult(True)


def _identity(value: Any) -> Any:
    return value


def test_kernel_spec_exposes_one_implementation_check():
    spec = KernelSpec(
        op="test.contract",
        backend="test",
        fn=_identity,
        priority=1,
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
    assert "availability" not in parameters
    assert "eligibility" not in parameters


def test_register_kernel_records_can_implement_callback():
    @register_kernel(
        op="test.register_can_implement",
        backend="test",
        priority=1,
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
