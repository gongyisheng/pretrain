from collections.abc import Callable
from typing import Any

from src.kernel.spec import (
    BuildMode,
    CanImplementFn,
    KernelSpec,
)


class KernelRegistry:
    def __init__(self):
        self._specs: dict[str, dict[str, KernelSpec]] = {}
        self._operation_can_implement: dict[str, CanImplementFn] = {}

    def register(self, spec: KernelSpec) -> None:
        by_backend = self._specs.setdefault(spec.op, {})
        if spec.backend in by_backend:
            raise ValueError(
                f"kernel already registered: op={spec.op!r}, backend={spec.backend!r}"
            )
        by_backend[spec.backend] = spec

    def implementations(self, op: str) -> tuple[KernelSpec, ...]:
        return tuple(self._specs.get(op, {}).values())

    def register_operation(self, op: str, can_implement: CanImplementFn) -> None:
        if op in self._operation_can_implement:
            raise ValueError(f"operation already registered: op={op!r}")
        self._operation_can_implement[op] = can_implement

    def operation_can_implement(self, op: str) -> CanImplementFn | None:
        return self._operation_can_implement.get(op)


KERNEL_REGISTRY = KernelRegistry()


def register_operation(op: str, can_implement: CanImplementFn) -> None:
    KERNEL_REGISTRY.register_operation(op, can_implement)


def register_kernel(
    op: str,
    backend: str,
    can_implement: CanImplementFn,
    build: BuildMode,
    autograd: bool,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        KERNEL_REGISTRY.register(
            KernelSpec(
                op=op,
                backend=backend,
                fn=fn,
                can_implement=can_implement,
                build=build,
                autograd=autograd,
            )
        )
        return fn

    return decorator
