from collections.abc import Mapping
from typing import Any

from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry
from src.kernel.spec import KernelSpec, SupportResult


class KernelSelectionError(RuntimeError):
    pass


def _support(
    spec: KernelSpec, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> SupportResult:
    availability = spec.availability()
    if not availability.supported:
        return availability
    return spec.eligibility(args, kwargs)


def _require_supported(
    spec: KernelSpec, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> None:
    support = _support(spec, args, kwargs)
    if not support.supported:
        raise KernelSelectionError(
            f"operation {spec.op!r} backend {spec.backend!r} is unavailable or "
            f"ineligible: {support.reason}"
        )


def select_kernel(
    op: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
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
        _require_supported(spec, args, kwargs)
        return spec

    rejected: list[str] = []
    for spec in sorted(specs, key=lambda item: item.priority, reverse=True):
        support = _support(spec, args, kwargs)
        if support.supported:
            return spec
        rejected.append(f"{spec.backend}: {support.reason}")
    raise KernelSelectionError(
        f"no eligible kernel for operation {op!r}; " + "; ".join(rejected)
    )


def dispatch(
    op: str,
    *args: Any,
    backend: str = "auto",
    registry: KernelRegistry = KERNEL_REGISTRY,
    **kwargs: Any,
) -> Any:
    selected = select_kernel(op, args, kwargs, backend=backend, registry=registry)
    return selected.fn(*args, **kwargs)
