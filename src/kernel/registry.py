from collections.abc import Callable, Mapping
from typing import Any

from src.kernel.spec import (
    BuildMode,
    CanImplementFn,
    KernelSpec,
    OperationSpec,
)


class KernelRegistry:
    def __init__(self):
        self._specs: dict[str, dict[str, KernelSpec]] = {}
        self._operations: dict[str, OperationSpec] = {}

    def register(self, spec: KernelSpec) -> None:
        by_backend = self._specs.get(spec.op)
        if by_backend is not None and spec.backend in by_backend:
            raise ValueError(
                f"kernel already registered: op={spec.op!r}, backend={spec.backend!r}"
            )
        operation = self._operations.get(spec.op)
        if operation is not None and not operation.supports(spec.backend):
            raise ValueError(
                f"kernel has no backend validator: "
                f"op={spec.op!r}, backend={spec.backend!r}"
            )
        if by_backend is None:
            self._specs[spec.op] = {spec.backend: spec}
        else:
            by_backend[spec.backend] = spec

    def implementations(self, op: str) -> tuple[KernelSpec, ...]:
        return tuple(self._specs.get(op, {}).values())

    def register_operation(
        self,
        op: str,
        can_implement: Mapping[str, CanImplementFn],
    ) -> None:
        if op in self._operations:
            raise ValueError(f"operation already registered: op={op!r}")
        operation = OperationSpec(can_implement)
        missing = [
            spec.backend
            for spec in self.implementations(op)
            if not operation.supports(spec.backend)
        ]
        if missing:
            backends = ", ".join(sorted(missing))
            raise ValueError(
                f"operation has no backend validator for registered kernels: "
                f"op={op!r}, backends={backends}"
            )
        self._operations[op] = operation

    def operation(self, op: str) -> OperationSpec | None:
        return self._operations.get(op)

    def can_implement(self, op: str, backend: str) -> CanImplementFn | None:
        operation = self.operation(op)
        if operation is None:
            return None
        return operation.validator(backend)


KERNEL_REGISTRY = KernelRegistry()


def register_operation(
    op: str,
    can_implement: Mapping[str, CanImplementFn],
) -> None:
    KERNEL_REGISTRY.register_operation(op, can_implement)


def register_kernel(
    op: str,
    backend: str,
    build: BuildMode,
    autograd: bool,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        KERNEL_REGISTRY.register(
            KernelSpec(
                op=op,
                backend=backend,
                fn=fn,
                build=build,
                autograd=autograd,
            )
        )
        return fn

    return decorator
