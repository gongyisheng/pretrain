import pytest
import torch

from src.kernel.registry import KernelRegistry
from src.kernel.selector import KernelSelectionError, select_kernel
from src.kernel.spec import CPU, KernelSpec, PlatformInfo, cuda

SM89 = PlatformInfo("cuda", (8, 9))
SM100 = PlatformInfo("cuda", (10, 0))
HOST = PlatformInfo()


def _spec(op, backend, capabilities=frozenset(), fn=None):
    return KernelSpec(
        op=op,
        backend=backend,
        fn=fn or (lambda *a, **k: backend),
        build="jit",
        autograd=False,
        capabilities=capabilities,
    )


def _registry(*specs):
    registry = KernelRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


def test_unregistered_op_raises():
    with pytest.raises(KernelSelectionError, match="no kernels registered"):
        select_kernel("gemm.missing", registry=_registry())


def test_explicit_backend_wins_without_consulting_capabilities():
    registry = _registry(_spec("test.op", "cublaslt", frozenset({cuda((99, 0))})))

    assert select_kernel("test.op", "cublaslt", registry=registry).backend == "cublaslt"


def test_explicit_unknown_backend_lists_what_is_registered():
    registry = _registry(_spec("test.op", "eager"), _spec("test.op", "triton"))

    with pytest.raises(KernelSelectionError, match="eager, triton"):
        select_kernel("test.op", "cublaslt", registry=registry)


def test_single_registration_returns_without_consulting_the_platform():
    registry = _registry(_spec("test.op", "cublaslt", frozenset({cuda((99, 0))})))

    assert select_kernel("test.op", registry=registry).backend == "cublaslt"


def test_multi_spec_without_a_platform_is_a_programming_error():
    registry = _registry(
        _spec("test.op", "eager", frozenset({CPU})),
        _spec("test.op", "triton", frozenset({cuda((8, 0))})),
    )

    with pytest.raises(KernelSelectionError, match="no platform was supplied"):
        select_kernel("test.op", registry=registry)


def test_disjoint_capabilities_resolve_to_one_backend():
    registry = _registry(
        _spec("test.op", "eager", frozenset({CPU})),
        _spec("test.op", "triton", frozenset({cuda((8, 0))})),
    )

    assert select_kernel("test.op", registry=registry, platform=HOST).backend == "eager"


def test_no_eligible_backend_names_the_platform():
    registry = _registry(
        _spec("test.op", "eager", frozenset({CPU})),
        _spec("test.op", "cublaslt", frozenset({cuda((10, 0), (11, 0))})),
    )

    with pytest.raises(KernelSelectionError, match=r"cuda \(8, 9\)"):
        select_kernel("test.op", registry=registry, platform=SM89)


def test_two_eligible_backends_demand_an_explicit_choice():
    registry = _registry(
        _spec("test.op", "triton", frozenset({cuda((8, 9))})),
        _spec("test.op", "cublaslt", frozenset({cuda((10, 0), (11, 0))})),
    )

    with pytest.raises(KernelSelectionError, match="pass backend="):
        select_kernel("test.op", registry=registry, platform=SM100)


def test_platform_for_caches_per_device(monkeypatch):
    from src.kernel import selector

    selector.clear_cache()
    calls = []

    def counting_capability(device_index):
        calls.append(device_index)
        return (8, 9)

    monkeypatch.setattr(torch.cuda, "get_device_capability", counting_capability)
    try:
        first = selector._platform_for("cuda", 0)
        second = selector._platform_for("cuda", 0)
    finally:
        selector.clear_cache()

    assert first == second == PlatformInfo("cuda", (8, 9))
    assert calls == [0]


def test_platform_for_cpu_never_touches_cuda(monkeypatch):
    from src.kernel import selector

    def boom(*_args, **_kwargs):
        raise AssertionError("a CPU device must not query CUDA device capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)

    assert selector._platform_for("cpu", None) == PlatformInfo("cpu")


def test_resolve_caches_the_callable_per_op_and_backend():
    import src.kernel.ops.gemm  # noqa: F401  populates the global registry

    from src.kernel import selector

    selector.clear_cache()
    try:
        first = selector._resolve("gemm.grouped", "eager", "cpu", None)
        second = selector._resolve("gemm.grouped", "eager", "cpu", None)
        info = selector._resolve.cache_info()
    finally:
        selector.clear_cache()

    assert first is second
    assert (info.hits, info.misses) == (1, 1)


def test_cpu_device_call_resolves_to_eager_even_when_cuda_is_visible():
    """Regression guard: the platform gate is per-call (from the op's own tensors),
    not a process-global fact. A CPU-device call must resolve to eager even when this
    process can also see a CUDA device -- resolving by process-global platform used to
    route CPU-tensor calls into the CUDA-only triton backend and crash."""
    import src.kernel.backends.eager.gemm as eager_gemm
    import src.kernel.ops.gemm  # noqa: F401  populates the global registry

    from src.kernel import selector

    selector.clear_cache()
    try:
        fn = selector._resolve("gemm.grouped", None, "cpu", None)
    finally:
        selector.clear_cache()

    assert fn is eager_gemm.grouped_gemm
