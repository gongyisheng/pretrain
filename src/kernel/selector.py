from collections.abc import Mapping
from typing import Any

import torch

from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry
from src.kernel.spec import BACKEND_PRIORITIES, CanImplementFn, CheckResult, KernelSpec


class KernelSelectionError(RuntimeError):
    pass


def _check(
    spec: KernelSpec,
    can_implement: CanImplementFn,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
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
    return can_implement(args, kwargs)


def _require_implementation(
    spec: KernelSpec,
    can_implement: CanImplementFn,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> None:
    result = _check(spec, can_implement, args, kwargs)
    if not result.ok:
        raise KernelSelectionError(
            f"operation {spec.op!r} backend {spec.backend!r} cannot implement "
            f"this call: {result.reason}"
        )


def _require_valid_contract(
    validate: CanImplementFn | None,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> None:
    if validate is None:
        return
    result = validate(args, kwargs)
    if not result.ok:
        raise KernelSelectionError(result.reason)


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

    operation = registry.operation(op)
    if operation is None:
        raise KernelSelectionError(
            f"operation {op!r} has no registered metadata; import its public "
            "src.kernel.ops entry point before dispatch"
        )
    if backend != "auto":
        matches = [spec for spec in specs if spec.backend == backend]
        if not matches:
            registered = ", ".join(sorted(spec.backend for spec in specs))
            raise KernelSelectionError(
                f"operation {op!r} has no backend {backend!r}; registered: {registered}"
            )
        spec = matches[0]
        can_implement = registry.can_implement(op, spec.backend)
        if can_implement is None:
            raise KernelSelectionError(
                f"operation {op!r} backend {spec.backend!r} has no registered validator"
            )
        _require_valid_contract(operation.validate, args, kwargs)
        _require_implementation(spec, can_implement, args, kwargs)
        return spec

    _require_valid_contract(operation.validate, args, kwargs)
    rejected: list[str] = []
    for spec in sorted(
        specs,
        key=lambda item: BACKEND_PRIORITIES[item.backend],
        reverse=True,
    ):
        can_implement = registry.can_implement(op, spec.backend)
        if can_implement is None:
            raise KernelSelectionError(
                f"operation {op!r} backend {spec.backend!r} has no registered validator"
            )
        result = _check(spec, can_implement, args, kwargs)
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
