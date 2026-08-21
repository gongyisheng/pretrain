import pytest

from src.kernel.registry import KernelRegistry, register_kernel
from src.kernel.spec import CPU, KernelSpec


def _fn():
    return "ran"


def _spec(**overrides):
    kwargs = dict(op="test.op", backend="triton", fn=_fn, build="jit", autograd=False)
    return KernelSpec(**(kwargs | overrides))


REGISTERED_KERNELS = [
    ("test.op", "cublaslt"),
    ("gemm.fp8_scaled_mm", "eager"),
    ("test.op", "triton"),
    ("test.op", "eager"),
]

LOOKUP_CASES = [
    ("test.op", ("cublaslt", "triton", "eager")),
    ("gemm.fp8_scaled_mm", ("eager",)),
    ("test.missing", ()),
]

REGISTER_KERNEL_CASES = [
    (
        dict(
            op="test.full",
            backend="eager",
            build="eager",
            autograd=True,
            capabilities=frozenset({CPU}),
            reference=True,
        ),
        KernelSpec(
            op="test.full",
            backend="eager",
            fn=_fn,
            build="eager",
            autograd=True,
            capabilities=frozenset({CPU}),
            reference=True,
        ),
    ),
    (
        dict(op="test.defaults", backend="triton", build="jit", autograd=False),
        KernelSpec(
            op="test.defaults",
            backend="triton",
            fn=_fn,
            build="jit",
            autograd=False,
            capabilities=frozenset(),
            reference=False,
        ),
    ),
]


@pytest.fixture
def registry(monkeypatch):
    """A fresh global registry, so decorator tests never leak across runs."""
    fresh = KernelRegistry()
    monkeypatch.setattr("src.kernel.registry.KERNEL_REGISTRY", fresh)
    return fresh


@pytest.mark.parametrize("op,expected", LOOKUP_CASES)
def test_kernel_registry_implementations(op, expected):
    registry = KernelRegistry()
    specs = {key: _spec(op=key[0], backend=key[1]) for key in REGISTERED_KERNELS}
    for spec in specs.values():
        registry.register(spec)

    assert registry.implementations(op) == tuple(
        specs[(op, backend)] for backend in expected
    )
    # pre-instance register, not pre-class
    assert KernelRegistry().implementations(op) == ()


def test_kernel_registery_register_raise_error():
    registry = KernelRegistry()
    registry.register(_spec())

    with pytest.raises(ValueError, match="kernel already registered"):
        registry.register(_spec())


@pytest.mark.parametrize("kwargs,expected", REGISTER_KERNEL_CASES)
def test_register_kernel(registry, kwargs, expected):
    assert register_kernel(**kwargs)(_fn) is _fn
    assert registry.implementations(kwargs["op"]) == (expected,)


def test_register_kernel_raise_error(registry):
    decorate = register_kernel(
        op="test.dup", backend="eager", build="eager", autograd=False
    )
    decorate(_fn)
    assert len(registry.implementations("test.dup")) == 1

    with pytest.raises(ValueError, match="kernel already registered"):
        decorate(_fn)
