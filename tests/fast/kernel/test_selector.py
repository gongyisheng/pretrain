from collections.abc import Callable, Mapping
from typing import Any

import pytest
import torch

from src.kernel.ops import gemm as gemm_module
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
    can_implement: Callable[[tuple[Any, ...], Mapping[str, Any]], CheckResult] = _yes,
    autograd: bool = True,
) -> KernelSpec:
    return KernelSpec(
        op="test.identity",
        backend=backend,
        fn=fn,
        can_implement=can_implement,
        build="eager",
        autograd=autograd,
    )


_selection_cache_calls = 0


def _count_selection_cache_checks(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    global _selection_cache_calls
    del args, kwargs
    _selection_cache_calls += 1
    return CheckResult(True)


KERNEL_REGISTRY.register(
    KernelSpec(
        op="test.selection_cache",
        backend="eager",
        fn=lambda value: value,
        can_implement=_count_selection_cache_checks,
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


def test_auto_cache_miss_runs_shared_and_backend_checks_once():
    calls = {"shared": 0, "backend": 0}

    def shared(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["shared"] += 1
        return CheckResult(True)

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["backend"] += 1
        return CheckResult(True)

    KERNEL_REGISTRY.register_operation("test.shared_cache_miss", shared)
    KERNEL_REGISTRY.register(
        KernelSpec(
            op="test.shared_cache_miss",
            backend="eager",
            fn=lambda value: value,
            can_implement=backend,
            build="eager",
            autograd=True,
        )
    )

    select_kernel("test.shared_cache_miss", (torch.ones(2),), {})

    assert calls == {"shared": 1, "backend": 1}


def test_auto_cache_hit_rechecks_shared_without_backend_check():
    calls = {"shared": 0, "backend": 0}

    def shared(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["shared"] += 1
        return CheckResult(True)

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["backend"] += 1
        return CheckResult(True)

    KERNEL_REGISTRY.register_operation("test.shared_cache_hit", shared)
    KERNEL_REGISTRY.register(
        KernelSpec(
            op="test.shared_cache_hit",
            backend="eager",
            fn=lambda value: value,
            can_implement=backend,
            build="eager",
            autograd=True,
        )
    )
    tensor = torch.ones(2)

    select_kernel("test.shared_cache_hit", (tensor,), {})
    select_kernel("test.shared_cache_hit", (tensor,), {})

    assert calls == {"shared": 2, "backend": 1}


def test_grouped_gemm_warm_cache_rechecks_shared_without_eager_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {"shared": 0, "backend": 0}
    operation = "gemm.grouped"
    shared = KERNEL_REGISTRY.operation_can_implement(operation)
    eager_spec = next(
        spec
        for spec in KERNEL_REGISTRY.implementations(operation)
        if spec.backend == "eager"
    )

    assert shared is not None

    def counted_shared(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        calls["shared"] += 1
        return shared(args, kwargs)

    def counted_eager(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        calls["backend"] += 1
        return eager_spec.can_implement(args, kwargs)

    monkeypatch.setattr(
        gemm_module, "can_implement_grouped_gemm_shared", counted_shared
    )
    monkeypatch.setitem(
        KERNEL_REGISTRY._operation_can_implement, operation, counted_shared
    )
    monkeypatch.setitem(
        KERNEL_REGISTRY._specs,
        operation,
        {
            "eager": KernelSpec(
                op=eager_spec.op,
                backend=eager_spec.backend,
                fn=eager_spec.fn,
                can_implement=counted_eager,
                build=eager_spec.build,
                autograd=eager_spec.autograd,
            )
        },
    )
    a = torch.ones(2, 3)
    b = torch.ones(3, 4)
    offs = torch.tensor([3], dtype=torch.int32)
    clear_selection_cache()

    try:
        gemm_module.grouped_gemm(a, b, offs)
        gemm_module.grouped_gemm(a, b, offs)
    finally:
        clear_selection_cache()

    assert calls == {"shared": 2, "backend": 1}


def test_explicit_backend_bypasses_warmed_auto_cache():
    calls: list[str] = []

    def shared(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls.append("shared")
        return CheckResult(True)

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls.append("backend")
        return CheckResult(True)

    KERNEL_REGISTRY.register_operation("test.explicit_warmed_cache", shared)
    KERNEL_REGISTRY.register(
        KernelSpec(
            op="test.explicit_warmed_cache",
            backend="eager",
            fn=lambda value: value,
            can_implement=backend,
            build="eager",
            autograd=True,
        )
    )
    tensor = torch.ones(2)

    select_kernel("test.explicit_warmed_cache", (tensor,), {})
    select_kernel("test.explicit_warmed_cache", (tensor,), {}, backend="eager")
    select_kernel("test.explicit_warmed_cache", (tensor,), {}, backend="eager")

    assert calls == ["shared", "backend", "shared", "backend", "shared", "backend"]
    assert calls.count("shared") == 3
    assert calls.count("backend") == 3


def test_shared_failure_rechecks_mutated_tensor_before_cached_selection():
    calls = {"shared": 0, "backend": 0}

    def shared(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del kwargs
        calls["shared"] += 1
        if bool(torch.all(args[0] >= 0)):
            return CheckResult(True)
        return CheckResult(False, "tensor values must be nonnegative")

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["backend"] += 1
        return CheckResult(True)

    KERNEL_REGISTRY.register_operation("test.shared_mutable_tensor", shared)
    KERNEL_REGISTRY.register(
        KernelSpec(
            op="test.shared_mutable_tensor",
            backend="eager",
            fn=lambda value: value,
            can_implement=backend,
            build="eager",
            autograd=True,
        )
    )
    tensor = torch.ones(2)

    select_kernel("test.shared_mutable_tensor", (tensor,), {})
    tensor[0] = -1
    with pytest.raises(KernelSelectionError) as error:
        select_kernel("test.shared_mutable_tensor", (tensor,), {})

    assert error.value.args == ("tensor values must be nonnegative",)
    assert str(error.value) == "tensor values must be nonnegative"
    assert calls == {"shared": 2, "backend": 1}


def test_shared_failure_does_not_populate_auto_selection_cache():
    calls = {"shared": 0, "backend": 0}
    is_valid = False

    def shared(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["shared"] += 1
        if is_valid:
            return CheckResult(True)
        return CheckResult(False, "shared call is invalid")

    def backend(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["backend"] += 1
        return CheckResult(True)

    KERNEL_REGISTRY.register_operation("test.shared_failure_cache", shared)
    KERNEL_REGISTRY.register(
        KernelSpec(
            op="test.shared_failure_cache",
            backend="eager",
            fn=lambda value: value,
            can_implement=backend,
            build="eager",
            autograd=True,
        )
    )
    tensor = torch.ones(2)

    with pytest.raises(KernelSelectionError) as error:
        select_kernel("test.shared_failure_cache", (tensor,), {})

    assert error.value.args == ("shared call is invalid",)
    assert str(error.value) == "shared call is invalid"
    is_valid = True
    select_kernel("test.shared_failure_cache", (tensor,), {})

    assert calls == {"shared": 2, "backend": 1}


def test_local_registry_without_shared_callback_selects_normally():
    registry = KernelRegistry()
    registry.register(_spec("triton", lambda value: value))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "triton"


def test_clear_selection_cache_forces_another_eligibility_check():
    tensor = torch.ones(2, 3)

    select_kernel("test.selection_cache", (tensor,), {})
    clear_selection_cache()
    select_kernel("test.selection_cache", (tensor,), {})

    assert _selection_cache_calls == 2


def test_auto_selects_highest_priority_eligible_backend():
    registry = KernelRegistry()
    registry.register(_spec("eager", lambda x: ("eager", x)))
    registry.register(_spec("cublaslt", lambda x: ("cublaslt", x)))
    registry.register(_spec("torch", lambda x: ("torch", x)))
    registry.register(_spec("triton", lambda x: ("triton", x)))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "cublaslt"


def test_auto_uses_torch_when_optimized_backend_is_ineligible():
    registry = KernelRegistry()
    registry.register(_spec("torch", lambda x: x))
    registry.register(_spec("triton", lambda x: x, can_implement=_no))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "torch"


def test_auto_runs_shared_once_before_falling_back_to_an_eligible_backend():
    calls = {"shared": 0, "rejector": 0, "fallback": 0}

    def shared(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["shared"] += 1
        return CheckResult(True)

    def rejector(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["rejector"] += 1
        return CheckResult(False, "higher-priority backend is ineligible")

    def fallback(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> CheckResult:
        del args, kwargs
        calls["fallback"] += 1
        return CheckResult(True)

    registry = KernelRegistry()
    registry.register_operation("test.identity", shared)
    registry.register(_spec("eager", lambda value: value, fallback))
    registry.register(_spec("triton", lambda value: value, rejector))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "eager"
    assert calls == {"shared": 1, "rejector": 1, "fallback": 1}


def test_forced_selection_uses_requested_backend():
    registry = KernelRegistry()
    registry.register(_spec("torch", lambda x: x))
    registry.register(_spec("triton", lambda x: x))

    selected = select_kernel("test.identity", (3,), {}, "torch", registry)

    assert selected.backend == "torch"


def test_forced_ineligible_backend_raises_reason():
    registry = KernelRegistry()
    registry.register(_spec("triton", lambda x: x, can_implement=_no))

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

    registry.register(_spec("torch", lambda x: x, reject_torch))
    registry.register(_spec("triton", lambda x: x, reject_triton))

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

    registry.register(_spec("triton", lambda x: x, can_implement, autograd=False))
    x = torch.ones(2, requires_grad=True)

    with pytest.raises(
        KernelSelectionError, match="does not support ordinary autograd"
    ):
        select_kernel("test.identity", (x,), {}, "triton", registry)

    assert calls == 0


def test_unknown_forced_backend_lists_registered_backends():
    registry = KernelRegistry()
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

    registry = KernelRegistry()
    registry.register(_spec("torch", fallback))
    registry.register(_spec("triton", defect))

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
    registry.register(_spec("triton", lambda value: value, can_implement=can_implement))

    with pytest.raises(KernelSelectionError, match="unsupported test call"):
        select_kernel("test.identity", (3,), {}, "triton", registry)

    assert calls == 1


def test_auto_falls_back_when_optimized_backend_cannot_build_autograd():
    registry = KernelRegistry()
    registry.register(_spec("torch", lambda x: x))
    registry.register(_spec("triton", lambda x: x, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    selected = select_kernel("test.identity", (x,), {}, registry=registry)

    assert selected.backend == "torch"


def test_forced_backend_rejects_unsafe_grad_enabled_call():
    registry = KernelRegistry()
    registry.register(_spec("triton", lambda x: x, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    with pytest.raises(
        KernelSelectionError,
        match="triton.*does not support ordinary autograd.*grad-enabled",
    ):
        select_kernel("test.identity", (x,), {}, "triton", registry)


def test_non_autograd_backend_is_selectable_without_grad_tracking():
    registry = KernelRegistry()
    registry.register(_spec("triton", lambda x: x, autograd=False))
    x = torch.ones(2, device="cpu", requires_grad=True)

    with torch.no_grad():
        selected = select_kernel("test.identity", (x,), {}, registry=registry)

    assert selected.backend == "triton"


def test_non_autograd_backend_is_selectable_inside_autograd_function_forward():
    registry = KernelRegistry()
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
    with pytest.raises(TypeError, match="autograd must be bool"):

        @register_kernel(
            op="test.invalid_autograd",
            backend="eager",
            can_implement=_yes,
            build="eager",
            autograd="invalid",
        )
        def invalid_kernel(x: Any) -> Any:
            return x
