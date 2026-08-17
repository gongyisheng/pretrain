from collections.abc import Callable, Mapping
from typing import Any

import pytest
import torch

from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry, register_kernel
from src.kernel.selector import (
    KernelSelectionError,
    dispatch,
    select_kernel,
)
from src.kernel.spec import CheckResult, KernelSpec


def _yes(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
    del args, kwargs
    return CheckResult(True)


def _no(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
    del args, kwargs
    return CheckResult(False, "unsupported dtype")


def _spec(
    backend: str,
    fn: Callable[..., Any],
    autograd: bool = True,
) -> KernelSpec:
    return KernelSpec(
        op="test.identity",
        backend=backend,
        fn=fn,
        build="eager",
        autograd=autograd,
    )


def _registry(
    can_implement: Mapping[
        str, Callable[[tuple[Any, ...], Mapping[str, Any]], CheckResult]
    ]
    | None = None,
) -> KernelRegistry:
    if can_implement is None:
        can_implement = {
            "eager": _yes,
            "triton": _yes,
            "cublaslt": _yes,
        }
    registry = KernelRegistry()
    registry.register_operation("test.identity", can_implement)
    return registry


def test_auto_runs_backend_check_once_per_selection():
    calls = 0

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        return CheckResult(True)

    op = "test.backend_check_once"
    registry = KernelRegistry()
    registry.register_operation(op, {"eager": backend})
    registry.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )

    select_kernel(op, (torch.ones(2),), {}, registry=registry)

    assert calls == 1


def test_auto_revalidates_backend_on_every_selection():
    calls = 0

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        return CheckResult(True)

    op = "test.backend_revalidation"
    registry = KernelRegistry()
    registry.register_operation(op, {"eager": backend})
    registry.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )
    tensor = torch.ones(2)

    select_kernel(op, (tensor,), {}, registry=registry)
    select_kernel(op, (tensor,), {}, registry=registry)

    assert calls == 2


def test_auto_contract_runs_once_before_backend_checks():
    calls: list[str] = []

    def validate(args, kwargs):
        del args, kwargs
        calls.append("contract")
        return CheckResult(True)

    def reject(args, kwargs):
        del args, kwargs
        calls.append("cublaslt")
        return CheckResult(False, "unsupported")

    def accept(args, kwargs):
        del args, kwargs
        calls.append("eager")
        return CheckResult(True)

    op = "test.contract_once"
    registry = KernelRegistry()
    registry.register_operation(op, {"cublaslt": reject, "eager": accept}, validate)
    registry.register(
        KernelSpec(
            op=op,
            backend="cublaslt",
            fn=lambda value: value,
            build="eager",
            autograd=True,
        )
    )
    registry.register(
        KernelSpec(
            op=op,
            backend="eager",
            fn=lambda value: value,
            build="eager",
            autograd=True,
        )
    )

    selected = select_kernel(op, (torch.ones(2),), {}, registry=registry)

    assert selected.backend == "eager"
    assert calls == ["contract", "cublaslt", "eager"]


def test_auto_revalidates_operation_contract_on_every_selection():
    calls = 0

    def validate(args, kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return CheckResult(True)

    op = "test.contract_revalidation"
    registry = KernelRegistry()
    registry.register_operation(op, {"eager": _yes}, validate)
    registry.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )
    tensor = torch.ones(2)

    select_kernel(op, (tensor,), {}, registry=registry)
    select_kernel(op, (tensor,), {}, registry=registry)

    assert calls == 2


def test_explicit_backend_revalidates_every_selection():
    calls: list[str] = []

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls.append("backend")
        return CheckResult(True)

    op = "test.explicit_revalidation"
    registry = KernelRegistry()
    registry.register_operation(op, {"eager": backend})
    registry.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )
    tensor = torch.ones(2)

    select_kernel(op, (tensor,), {}, registry=registry)
    select_kernel(op, (tensor,), {}, backend="eager", registry=registry)
    select_kernel(op, (tensor,), {}, backend="eager", registry=registry)

    assert calls == ["backend", "backend", "backend"]


def test_auto_rechecks_value_dependent_validation():
    calls = 0

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del kwargs
        calls += 1
        if bool(torch.all(args[0] >= 0)):
            return CheckResult(True)
        return CheckResult(False, "tensor values must be nonnegative")

    op = "test.value_dependent_validator"
    registry = KernelRegistry()
    registry.register_operation(op, {"eager": backend})
    registry.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )
    tensor = torch.ones(2)

    select_kernel(op, (tensor,), {}, registry=registry)
    tensor[0] = -1
    with pytest.raises(KernelSelectionError, match="tensor values must be nonnegative"):
        select_kernel(op, (tensor,), {}, registry=registry)

    assert calls == 2


def test_auto_rechecks_backend_after_failure():
    calls = 0
    is_valid = False

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        if is_valid:
            return CheckResult(True)
        return CheckResult(False, "backend call is invalid")

    op = "test.backend_failure_recheck"
    registry = KernelRegistry()
    registry.register_operation(op, {"eager": backend})
    registry.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )
    tensor = torch.ones(2)

    with pytest.raises(KernelSelectionError, match="backend call is invalid"):
        select_kernel(op, (tensor,), {}, registry=registry)

    is_valid = True
    select_kernel(op, (tensor,), {}, registry=registry)

    assert calls == 2


def test_selection_requires_operation_metadata():
    registry = KernelRegistry()
    registry.register(_spec("triton", lambda value: value))

    with pytest.raises(KernelSelectionError, match="src.kernel.ops"):
        select_kernel("test.identity", (3,), {}, registry=registry)


def test_auto_selects_highest_priority_eligible_backend():
    registry = _registry()
    registry.register(_spec("eager", lambda x: ("eager", x)))
    registry.register(_spec("cublaslt", lambda x: ("cublaslt", x)))
    registry.register(_spec("triton", lambda x: ("triton", x)))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "cublaslt"


def test_auto_uses_eager_when_optimized_backend_is_ineligible():
    registry = _registry({"eager": _yes, "triton": _no})
    registry.register(_spec("eager", lambda x: x))
    registry.register(_spec("triton", lambda x: x))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "eager"


def test_auto_runs_validators_before_falling_back_to_an_eligible_backend():
    calls = {"rejector": 0, "fallback": 0}

    def rejector(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["rejector"] += 1
        return CheckResult(False, "higher-priority backend is ineligible")

    def fallback(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["fallback"] += 1
        return CheckResult(True)

    registry = _registry({"eager": fallback, "triton": rejector})
    registry.register(_spec("eager", lambda value: value))
    registry.register(_spec("triton", lambda value: value))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "eager"
    assert calls == {"rejector": 1, "fallback": 1}


def test_forced_selection_uses_requested_backend():
    registry = _registry({"eager": _yes, "triton": _yes})
    registry.register(_spec("eager", lambda x: x))
    registry.register(_spec("triton", lambda x: x))

    selected = select_kernel("test.identity", (3,), {}, "eager", registry)

    assert selected.backend == "eager"


def test_forced_ineligible_backend_raises_reason():
    registry = _registry({"triton": _no})
    registry.register(_spec("triton", lambda x: x))

    with pytest.raises(
        KernelSelectionError, match="test.identity.*triton.*unsupported dtype"
    ):
        select_kernel("test.identity", (3,), {}, "triton", registry)


def test_auto_calls_can_implement_once_per_candidate_and_aggregates_reasons():
    calls = {"eager": 0, "triton": 0}

    def reject_eager(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["eager"] += 1
        return CheckResult(False, "unsupported fallback")

    def reject_triton(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["triton"] += 1
        return CheckResult(False, "requires SM90")

    registry = _registry({"eager": reject_eager, "triton": reject_triton})
    registry.register(_spec("eager", lambda x: x))
    registry.register(_spec("triton", lambda x: x))

    with pytest.raises(
        KernelSelectionError,
        match="triton: requires SM90; eager: unsupported fallback",
    ):
        select_kernel("test.identity", (3,), {}, registry=registry)

    assert calls == {"eager": 1, "triton": 1}


def test_autograd_gate_runs_before_backend_validation():
    calls = 0

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        return CheckResult(True)

    registry = _registry({"triton": backend})
    registry.register(_spec("triton", lambda x: x, autograd=False))
    x = torch.ones(2, requires_grad=True)

    with pytest.raises(
        KernelSelectionError, match="does not support ordinary autograd"
    ):
        select_kernel("test.identity", (x,), {}, "triton", registry)

    assert calls == 0


def test_unknown_forced_backend_lists_registered_backends():
    registry = _registry({"eager": _yes, "triton": _yes})
    registry.register(_spec("eager", lambda x: x))
    registry.register(_spec("triton", lambda x: x))

    with pytest.raises(
        KernelSelectionError,
        match="test.identity.*cuda.*registered: eager, triton",
    ):
        select_kernel("test.identity", (3,), {}, "cuda", registry)


def test_dispatch_does_not_retry_after_selected_kernel_raises():
    calls = {"eager": 0, "triton": 0}

    def fallback(x: Any) -> Any:
        calls["eager"] += 1
        return x

    def defect(x: Any) -> Any:
        del x
        calls["triton"] += 1
        raise RuntimeError("kernel defect")

    registry = _registry({"eager": _yes, "triton": _yes})
    registry.register(_spec("eager", fallback))
    registry.register(_spec("triton", defect))

    with pytest.raises(RuntimeError, match="kernel defect"):
        dispatch("test.identity", (3,), {}, registry=registry)

    assert calls == {"eager": 0, "triton": 1}


def test_forced_selection_calls_backend_validation_exactly_once():
    calls = 0

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        return CheckResult(False, "unsupported test call")

    registry = _registry({"triton": backend})
    registry.register(_spec("triton", lambda value: value))

    with pytest.raises(KernelSelectionError, match="unsupported test call"):
        select_kernel("test.identity", (3,), {}, "triton", registry)

    assert calls == 1


def test_auto_falls_back_when_optimized_backend_cannot_build_autograd():
    registry = _registry({"eager": _yes, "triton": _yes})
    registry.register(_spec("eager", lambda x: x))
    registry.register(_spec("triton", lambda x: x, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    selected = select_kernel("test.identity", (x,), {}, registry=registry)

    assert selected.backend == "eager"


def test_forced_backend_rejects_unsafe_grad_enabled_call():
    registry = _registry({"triton": _yes})
    registry.register(_spec("triton", lambda x: x, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    with pytest.raises(
        KernelSelectionError,
        match="triton.*does not support ordinary autograd.*grad-enabled",
    ):
        select_kernel("test.identity", (x,), {}, "triton", registry)


def test_non_autograd_backend_is_selectable_without_grad_tracking():
    registry = _registry({"triton": _yes})
    registry.register(_spec("triton", lambda x: x, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    with torch.no_grad():
        selected = select_kernel("test.identity", (x,), {}, registry=registry)

    assert selected.backend == "triton"


def test_non_autograd_backend_is_selectable_inside_autograd_function_forward():
    registry = _registry({"triton": _yes})
    registry.register(_spec("triton", lambda x: x * 2, autograd=False))

    class ComposedIdentity(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
            del ctx
            return dispatch("test.identity", (x,), {}, registry=registry)

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            del ctx
            return grad_output * 2

    x = torch.ones(2, device="cpu", requires_grad=True)
    ComposedIdentity.apply(x).sum().backward()

    torch.testing.assert_close(x.grad, torch.full_like(x, 2))


@pytest.mark.parametrize("autograd", ["invalid", None, 1])
def test_kernel_spec_rejects_non_bool_autograd(autograd: Any):
    with pytest.raises(TypeError, match="autograd must be bool"):
        _spec("eager", lambda x: x, autograd=autograd)


def test_register_kernel_rejects_non_bool_autograd():
    op = "test.invalid_autograd"
    KERNEL_REGISTRY.register_operation(op, {"eager": _yes})

    with pytest.raises(TypeError, match="autograd must be bool"):

        @register_kernel(op=op, backend="eager", build="eager", autograd="invalid")
        def invalid_kernel(x: Any) -> Any:
            return x
