from collections.abc import Callable, Mapping
from typing import Any

import pytest
import torch

from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry, register_kernel
from src.kernel.selector import (
    KernelSelectionError,
    clear_selection_cache,
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
            "torch": _yes,
            "triton": _yes,
            "cublaslt": _yes,
        }
    registry = KernelRegistry()
    registry.register_operation("test.identity", can_implement)
    return registry


_selection_cache_calls = 0


def _count_selection_cache_checks(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    global _selection_cache_calls
    del args, kwargs
    _selection_cache_calls += 1
    return CheckResult(True)


KERNEL_REGISTRY.register_operation(
    "test.selection_cache", {"eager": _count_selection_cache_checks}
)
KERNEL_REGISTRY.register(
    KernelSpec(
        op="test.selection_cache",
        backend="eager",
        fn=lambda value: value,
        build="eager",
        autograd=True,
    )
)


@pytest.fixture(autouse=True)
def _reset_selection_cache() -> None:
    global _selection_cache_calls
    clear_selection_cache()
    _selection_cache_calls = 0


def test_selection_cache_reuses_default_auto_selection_for_identical_calls():
    tensor = torch.ones(2, 3)

    first = select_kernel("test.selection_cache", (tensor, 4), {"scale": 0.5})
    second = select_kernel("test.selection_cache", (tensor, 4), {"scale": 0.5})

    assert first is second
    assert _selection_cache_calls == 1


def test_selection_cache_reuses_normalized_dtype_arguments():
    tensor = torch.ones(2, 3)
    normalized_args = (tensor, torch.bfloat16, torch.float32)

    select_kernel("test.selection_cache", normalized_args, {})
    select_kernel("test.selection_cache", normalized_args, {})

    assert _selection_cache_calls == 1


def test_selection_cache_rechecks_auto_selection_when_tensor_stride_changes():
    contiguous = torch.ones(2, 3)
    transposed = contiguous.t()

    select_kernel("test.selection_cache", (contiguous,), {})
    select_kernel("test.selection_cache", (transposed,), {})

    assert _selection_cache_calls == 2


def test_selection_cache_bypasses_explicit_backend_selection():
    tensor = torch.ones(2, 3)

    select_kernel("test.selection_cache", (tensor,), {}, backend="eager")
    select_kernel("test.selection_cache", (tensor,), {}, backend="eager")

    assert _selection_cache_calls == 2


def test_selection_cache_bypasses_compilation():
    tensor = torch.ones(2, 3)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        select_kernel("test.selection_cache", (tensor,), {})
        select_kernel("test.selection_cache", (tensor,), {})

    assert _selection_cache_calls == 2


def test_auto_cache_miss_runs_backend_check_once():
    calls = 0

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        return CheckResult(True)

    op = "test.cache_miss"
    KERNEL_REGISTRY.register_operation(op, {"eager": backend})
    KERNEL_REGISTRY.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )

    select_kernel(op, (torch.ones(2),), {})

    assert calls == 1


def test_auto_cache_hit_skips_backend_validation():
    calls = 0

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        return CheckResult(True)

    op = "test.cache_hit"
    KERNEL_REGISTRY.register_operation(op, {"eager": backend})
    KERNEL_REGISTRY.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )
    tensor = torch.ones(2)

    select_kernel(op, (tensor,), {})
    select_kernel(op, (tensor,), {})

    assert calls == 1


def test_explicit_backend_bypasses_warmed_auto_cache():
    calls: list[str] = []

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls.append("backend")
        return CheckResult(True)

    op = "test.explicit_warmed_cache"
    KERNEL_REGISTRY.register_operation(op, {"eager": backend})
    KERNEL_REGISTRY.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )
    tensor = torch.ones(2)

    select_kernel(op, (tensor,), {})
    select_kernel(op, (tensor,), {}, backend="eager")
    select_kernel(op, (tensor,), {}, backend="eager")

    assert calls == ["backend", "backend", "backend"]


def test_cache_hit_does_not_rerun_value_dependent_validation():
    calls = 0

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del kwargs
        calls += 1
        if bool(torch.all(args[0] >= 0)):
            return CheckResult(True)
        return CheckResult(False, "tensor values must be nonnegative")

    op = "test.value_dependent_validator"
    KERNEL_REGISTRY.register_operation(op, {"eager": backend})
    KERNEL_REGISTRY.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )
    tensor = torch.ones(2)

    select_kernel(op, (tensor,), {})
    tensor[0] = -1
    select_kernel(op, (tensor,), {})

    assert calls == 1


def test_backend_failure_does_not_populate_auto_selection_cache():
    calls = 0
    is_valid = False

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        nonlocal calls
        del args, kwargs
        calls += 1
        if is_valid:
            return CheckResult(True)
        return CheckResult(False, "backend call is invalid")

    op = "test.backend_failure_cache"
    KERNEL_REGISTRY.register_operation(op, {"eager": backend})
    KERNEL_REGISTRY.register(
        KernelSpec(
            op=op, backend="eager", fn=lambda value: value, build="eager", autograd=True
        )
    )
    tensor = torch.ones(2)

    with pytest.raises(KernelSelectionError, match="backend call is invalid"):
        select_kernel(op, (tensor,), {})

    is_valid = True
    select_kernel(op, (tensor,), {})

    assert calls == 2


def test_selection_requires_operation_metadata():
    registry = KernelRegistry()
    registry.register(_spec("triton", lambda value: value))

    with pytest.raises(KernelSelectionError, match="src.kernel.ops"):
        select_kernel("test.identity", (3,), {}, registry=registry)


def test_clear_selection_cache_forces_another_eligibility_check():
    tensor = torch.ones(2, 3)

    select_kernel("test.selection_cache", (tensor,), {})
    clear_selection_cache()
    select_kernel("test.selection_cache", (tensor,), {})

    assert _selection_cache_calls == 2


def test_auto_selects_highest_priority_eligible_backend():
    registry = _registry()
    registry.register(_spec("eager", lambda x: ("eager", x)))
    registry.register(_spec("cublaslt", lambda x: ("cublaslt", x)))
    registry.register(_spec("torch", lambda x: ("torch", x)))
    registry.register(_spec("triton", lambda x: ("triton", x)))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "cublaslt"


def test_auto_uses_torch_when_optimized_backend_is_ineligible():
    registry = _registry({"torch": _yes, "triton": _no})
    registry.register(_spec("torch", lambda x: x))
    registry.register(_spec("triton", lambda x: x))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "torch"


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
    registry = _registry({"torch": _yes, "triton": _yes})
    registry.register(_spec("torch", lambda x: x))
    registry.register(_spec("triton", lambda x: x))

    selected = select_kernel("test.identity", (3,), {}, "torch", registry)

    assert selected.backend == "torch"


def test_forced_ineligible_backend_raises_reason():
    registry = _registry({"triton": _no})
    registry.register(_spec("triton", lambda x: x))

    with pytest.raises(
        KernelSelectionError, match="test.identity.*triton.*unsupported dtype"
    ):
        select_kernel("test.identity", (3,), {}, "triton", registry)


def test_auto_calls_can_implement_once_per_candidate_and_aggregates_reasons():
    calls = {"torch": 0, "triton": 0}

    def reject_torch(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["torch"] += 1
        return CheckResult(False, "requires callable grouped_mm")

    def reject_triton(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["triton"] += 1
        return CheckResult(False, "requires SM90")

    registry = _registry({"torch": reject_torch, "triton": reject_triton})
    registry.register(_spec("torch", lambda x: x))
    registry.register(_spec("triton", lambda x: x))

    with pytest.raises(
        KernelSelectionError,
        match="triton: requires SM90; torch: requires callable grouped_mm",
    ):
        select_kernel("test.identity", (3,), {}, registry=registry)

    assert calls == {"torch": 1, "triton": 1}


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
    registry = _registry({"torch": _yes, "triton": _yes})
    registry.register(_spec("torch", lambda x: x))
    registry.register(_spec("triton", lambda x: x))

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

    registry = _registry({"torch": _yes, "triton": _yes})
    registry.register(_spec("torch", fallback))
    registry.register(_spec("triton", defect))

    with pytest.raises(RuntimeError, match="kernel defect"):
        dispatch("test.identity", (3,), {}, registry=registry)

    assert calls == {"torch": 0, "triton": 1}


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
    registry = _registry({"torch": _yes, "triton": _yes})
    registry.register(_spec("torch", lambda x: x))
    registry.register(_spec("triton", lambda x: x, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    selected = select_kernel("test.identity", (x,), {}, registry=registry)

    assert selected.backend == "torch"


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
