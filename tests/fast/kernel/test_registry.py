import pytest

from src.kernel.registry import KernelRegistry, register_kernel
from src.kernel.spec import CPU, KernelSpec, cuda


def _spec(op="test.op", backend="triton", capabilities=frozenset()):
    return KernelSpec(
        op=op,
        backend=backend,
        fn=lambda *a, **k: None,
        build="jit",
        autograd=False,
        capabilities=capabilities,
    )


def test_register_then_look_up_implementations():
    registry = KernelRegistry()
    spec = _spec()
    registry.register(spec)

    assert registry.implementations("test.op") == (spec,)


def test_implementations_is_empty_for_an_unknown_op():
    assert KernelRegistry().implementations("test.missing") == ()


def test_duplicate_op_and_backend_is_rejected():
    registry = KernelRegistry()
    registry.register(_spec())

    with pytest.raises(ValueError, match="kernel already registered"):
        registry.register(_spec())


def test_one_function_registers_under_several_ops():
    registry = KernelRegistry()

    def kernel():
        return None

    for op in ("gemm.int8_scaled_mm", "gemm.fp8_scaled_mm"):
        registry.register(
            KernelSpec(
                op=op,
                backend="triton",
                fn=kernel,
                build="jit",
                autograd=False,
                capabilities=frozenset({cuda((8, 0))}),
            )
        )

    assert registry.implementations("gemm.int8_scaled_mm")[0].fn is kernel
    assert registry.implementations("gemm.fp8_scaled_mm")[0].fn is kernel


def test_register_kernel_decorator_carries_capabilities():
    @register_kernel(
        op="test.decorated",
        backend="eager",
        build="eager",
        autograd=True,
        capabilities=frozenset({CPU}),
    )
    def kernel():
        return "ran"

    from src.kernel.registry import KERNEL_REGISTRY

    spec = KERNEL_REGISTRY.implementations("test.decorated")[0]

    assert spec.capabilities == frozenset({CPU})
    assert kernel() == "ran"
