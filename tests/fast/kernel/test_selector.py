from collections.abc import Callable, Mapping
from typing import Any

import pytest
import torch

from src.kernel.registry import KernelRegistry, register_kernel
from src.kernel.selector import KernelSelectionError, dispatch, select_kernel
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
    priority: int,
    can_implement: Callable[[tuple[Any, ...], Mapping[str, Any]], CheckResult] = _yes,
    autograd: bool = True,
) -> KernelSpec:
    return KernelSpec(
        op="test.identity",
        backend=backend,
        fn=fn,
        priority=priority,
        can_implement=can_implement,
        build="eager",
        autograd=autograd,
    )


def test_auto_selects_highest_priority_eligible_backend():
    registry = KernelRegistry()
    registry.register(_spec("torch", lambda x: ("torch", x), -100))
    registry.register(_spec("triton", lambda x: ("triton", x), 100))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "triton"


def test_auto_uses_torch_when_optimized_backend_is_ineligible():
    registry = KernelRegistry()
    registry.register(_spec("torch", lambda x: x, -100))
    registry.register(_spec("triton", lambda x: x, 100, can_implement=_no))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "torch"


def test_forced_selection_uses_requested_backend():
    registry = KernelRegistry()
    registry.register(_spec("torch", lambda x: x, -100))
    registry.register(_spec("triton", lambda x: x, 100))

    selected = select_kernel("test.identity", (3,), {}, "torch", registry)

    assert selected.backend == "torch"


def test_forced_ineligible_backend_raises_reason():
    registry = KernelRegistry()
    registry.register(_spec("triton", lambda x: x, 100, can_implement=_no))

    with pytest.raises(
        KernelSelectionError, match="test.identity.*triton.*unsupported dtype"
    ):
        select_kernel("test.identity", (3,), {}, "triton", registry)


def test_auto_calls_can_implement_once_per_candidate_and_aggregates_reasons():
    registry = KernelRegistry()
    calls = {"torch": 0, "triton": 0}

    def reject_torch(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["torch"] += 1
        return CheckResult(False, "requires callable grouped_mm")

    def reject_triton(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["triton"] += 1
        return CheckResult(False, "requires SM90")

    registry.register(_spec("torch", lambda x: x, 50, reject_torch))
    registry.register(_spec("triton", lambda x: x, 100, reject_triton))

    with pytest.raises(
        KernelSelectionError,
        match="triton: requires SM90; torch: requires callable grouped_mm",
    ):
        select_kernel("test.identity", (3,), {}, registry=registry)

    assert calls == {"torch": 1, "triton": 1}


def test_autograd_gate_runs_before_can_implement():
    registry = KernelRegistry()
    calls = 0

    def can_implement(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        return CheckResult(True)

    registry.register(
        _spec("optimized", lambda x: x, 100, can_implement, autograd=False)
    )
    x = torch.ones(2, requires_grad=True)

    with pytest.raises(
        KernelSelectionError, match="does not support ordinary autograd"
    ):
        select_kernel("test.identity", (x,), {}, "optimized", registry)

    assert calls == 0


def test_duplicate_registration_raises_value_error():
    registry = KernelRegistry()
    spec = _spec("torch", lambda x: x, -100)
    registry.register(spec)

    with pytest.raises(ValueError, match="kernel already registered"):
        registry.register(spec)


def test_unknown_forced_backend_lists_registered_backends():
    registry = KernelRegistry()
    registry.register(_spec("torch", lambda x: x, -100))
    registry.register(_spec("triton", lambda x: x, 100))

    with pytest.raises(
        KernelSelectionError,
        match="test.identity.*cuda.*registered: torch, triton",
    ):
        select_kernel("test.identity", (3,), {}, "cuda", registry)


def test_dispatch_does_not_retry_after_selected_kernel_raises():
    calls = {"torch": 0, "triton": 0}

    def fallback(x: Any) -> Any:
        calls["torch"] += 1
        return x

    def defect(x: Any) -> Any:
        del x
        calls["triton"] += 1
        raise RuntimeError("kernel defect")

    registry = KernelRegistry()
    registry.register(_spec("torch", fallback, -100))
    registry.register(_spec("triton", defect, 100))

    with pytest.raises(RuntimeError, match="kernel defect"):
        dispatch("test.identity", (3,), {}, registry=registry)

    assert calls == {"torch": 0, "triton": 1}


def test_forced_selection_calls_can_implement_exactly_once():
    calls = 0

    def can_implement(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        return CheckResult(False, "unsupported test call")

    registry = KernelRegistry()
    registry.register(
        _spec("triton", lambda value: value, 100, can_implement=can_implement)
    )

    with pytest.raises(KernelSelectionError, match="unsupported test call"):
        select_kernel("test.identity", (3,), {}, "triton", registry)

    assert calls == 1


def test_auto_falls_back_when_optimized_backend_cannot_build_autograd():
    registry = KernelRegistry()
    registry.register(_spec("torch", lambda x: x, -100))
    registry.register(_spec("optimized", lambda x: x, 100, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    selected = select_kernel("test.identity", (x,), {}, registry=registry)

    assert selected.backend == "torch"


def test_forced_backend_rejects_unsafe_grad_enabled_call():
    registry = KernelRegistry()
    registry.register(_spec("optimized", lambda x: x, 100, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    with pytest.raises(
        KernelSelectionError,
        match="optimized.*does not support ordinary autograd.*grad-enabled",
    ):
        select_kernel("test.identity", (x,), {}, "optimized", registry)


def test_non_autograd_backend_is_selectable_without_grad_tracking():
    registry = KernelRegistry()
    registry.register(_spec("optimized", lambda x: x, 100, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    with torch.no_grad():
        selected = select_kernel("test.identity", (x,), {}, registry=registry)

    assert selected.backend == "optimized"


def test_non_autograd_backend_is_selectable_inside_autograd_function_forward():
    registry = KernelRegistry()
    registry.register(_spec("optimized", lambda x: x * 2, 100, autograd=False))

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
        _spec("invalid", lambda x: x, 0, autograd=autograd)


def test_register_kernel_rejects_non_bool_autograd():
    with pytest.raises(TypeError, match="autograd must be bool"):

        @register_kernel(
            op="test.invalid_autograd",
            backend="invalid",
            priority=0,
            can_implement=_yes,
            build="eager",
            autograd="invalid",
        )
        def invalid_kernel(x: Any) -> Any:
            return x
