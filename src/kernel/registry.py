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

    def register(self, spec: KernelSpec) -> None:
        by_backend = self._specs.setdefault(spec.op, {})
        if spec.backend in by_backend:
            raise ValueError(
                f"kernel already registered: op={spec.op!r}, backend={spec.backend!r}"
            )
        by_backend[spec.backend] = spec

    def implementations(self, op: str) -> tuple[KernelSpec, ...]:
        return tuple(self._specs.get(op, {}).values())


KERNEL_REGISTRY = KernelRegistry()


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
