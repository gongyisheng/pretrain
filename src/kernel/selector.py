from collections.abc import Mapping
from typing import Any

import torch

from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry
from src.kernel.spec import BACKEND_PRIORITIES, CheckResult, KernelSpec


class KernelSelectionError(RuntimeError):
    pass


def _check(
    spec: KernelSpec, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    tensor_requires_grad = any(
        isinstance(value, torch.Tensor) and value.requires_grad for value in args
    ) or any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in kwargs.values()
    )
    if not spec.autograd and torch.is_grad_enabled() and tensor_requires_grad:
        return CheckResult(
            False,
            "backend does not support ordinary autograd for a grad-enabled call "
            "with tensor inputs requiring grad",
        )
    return spec.can_implement(args, kwargs)


def _require_implementation(
    spec: KernelSpec, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> None:
    result = _check(spec, args, kwargs)
    if not result.ok:
        raise KernelSelectionError(
            f"operation {spec.op!r} backend {spec.backend!r} cannot implement "
            f"this call: {result.reason}"
        )


def select_kernel(
    op: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    backend: str = "auto",
    registry: KernelRegistry = KERNEL_REGISTRY,
) -> KernelSpec:
    specs = registry.implementations(op)
    if not specs:
        raise KernelSelectionError(f"no kernels registered for operation {op!r}")

    if backend != "auto":
        matches = [spec for spec in specs if spec.backend == backend]
        if not matches:
            registered = ", ".join(sorted(spec.backend for spec in specs))
            raise KernelSelectionError(
                f"operation {op!r} has no backend {backend!r}; registered: {registered}"
            )
        spec = matches[0]
        _require_implementation(spec, args, kwargs)
        return spec

    rejected: list[str] = []
    for spec in sorted(
        specs,
        key=lambda item: BACKEND_PRIORITIES[item.backend],
        reverse=True,
    ):
        result = _check(spec, args, kwargs)
        if result.ok:
            return spec
        rejected.append(f"{spec.backend}: {result.reason}")
    raise KernelSelectionError(
        f"no eligible kernel for operation {op!r}; " + "; ".join(rejected)
    )


def dispatch(
    op: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    backend: str = "auto",
    registry: KernelRegistry = KERNEL_REGISTRY,
) -> Any:
    selected = select_kernel(op, args, kwargs, backend, registry)
    return selected.fn(*args, **kwargs)
