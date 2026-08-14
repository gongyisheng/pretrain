import pytest

from src.kernel.registry import KernelRegistry
from src.kernel.selector import KernelSelectionError, dispatch, select_kernel
from src.kernel.spec import KernelSpec, SupportResult


def _yes(*_args, **_kwargs):
    return SupportResult(True)


def _no(*_args, **_kwargs):
    return SupportResult(False, "unsupported dtype")


def _spec(backend, fn, priority, eligibility=_yes):
    return KernelSpec(
        op="test.identity",
        backend=backend,
        fn=fn,
        priority=priority,
        availability=lambda: SupportResult(True),
        eligibility=eligibility,
        build="eager",
        autograd="native",
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
    registry.register(_spec("triton", lambda x: x, 100, eligibility=_no))

    selected = select_kernel("test.identity", (3,), {}, registry=registry)

    assert selected.backend == "torch"


def test_forced_selection_uses_requested_backend():
    registry = KernelRegistry()
    registry.register(_spec("torch", lambda x: x, -100))
    registry.register(_spec("triton", lambda x: x, 100))

    selected = select_kernel(
        "test.identity", (3,), {}, backend="torch", registry=registry
    )

    assert selected.backend == "torch"


def test_forced_ineligible_backend_raises_reason():
    registry = KernelRegistry()
    registry.register(_spec("triton", lambda x: x, 100, eligibility=_no))

    with pytest.raises(
        KernelSelectionError, match="test.identity.*triton.*unsupported dtype"
    ):
        select_kernel("test.identity", (3,), {}, backend="triton", registry=registry)


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
        select_kernel("test.identity", (3,), {}, backend="cuda", registry=registry)


def test_dispatch_does_not_retry_after_selected_kernel_raises():
    calls = {"torch": 0, "triton": 0}

    def fallback(x):
        calls["torch"] += 1
        return x

    def defect(_x):
        calls["triton"] += 1
        raise RuntimeError("kernel defect")

    registry = KernelRegistry()
    registry.register(_spec("torch", fallback, -100))
    registry.register(_spec("triton", defect, 100))

    with pytest.raises(RuntimeError, match="kernel defect"):
        dispatch("test.identity", 3, registry=registry)

    assert calls == {"torch": 0, "triton": 1}
