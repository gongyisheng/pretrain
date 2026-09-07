from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable

import torch

from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry
from src.kernel.spec import KernelSpec, PlatformInfo


class KernelSelectionError(RuntimeError):
    pass


def compile_safe_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Unbounded cache Dynamo can trace: it reads the dict rather than being
    ignored and re-tracing the body, as it does for `functools.lru_cache`."""
    cache: dict[tuple[Any, ...], Any] = {}

    @wraps(fn)
    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]

    wrapper.cache_clear = cache.clear
    return wrapper


@compile_safe_cache
def _platform_for(device_type: str, device_index: int | None) -> PlatformInfo:
    if device_type != "cuda":
        return PlatformInfo(device_type)
    return PlatformInfo("cuda", torch.cuda.get_device_capability(device_index))


def select_kernel(
    op: str,
    backend: str | None = None,
    platform: PlatformInfo | None = None,
    registry: KernelRegistry = KERNEL_REGISTRY,
) -> KernelSpec:
    """Resolve `op` to one implementation. A hard eligibility gate, not a ranking."""

    # find registered implementations or honor an explicit backend choice.
    specs = registry.implementations(op)
    if not specs:
        raise KernelSelectionError(f"no kernels registered for operation {op!r}")
    registered = ", ".join(sorted(spec.backend for spec in specs))
    if backend is not None:
        for spec in specs:
            if spec.backend == backend:
                return spec
        raise KernelSelectionError(
            f"operation {op!r} has no backend {backend!r}; registered: {registered}"
        )

    # no selection is needed when there is only one implementation.
    if len(specs) == 1:
        return specs[0]

    # use the platform to filter implementations by availability.
    if platform is None:
        raise KernelSelectionError(
            f"operation {op!r} has multiple backends ({registered}) and no "
            "platform was supplied to choose among them"
        )
    eligible = [spec for spec in specs if spec.is_available(platform)]

    # prefer one optimized implementation, then fall back to a reference.
    optimized = [spec for spec in eligible if not spec.reference]
    if len(optimized) == 1:
        return optimized[0]
    if not optimized:
        reference = [spec for spec in eligible if spec.reference]
        if len(reference) == 1:
            return reference[0]
        if not reference:
            raise KernelSelectionError(
                f"operation {op!r} has no backend for {platform.device_type} "
                f"{platform.arch}; registered: {registered}"
            )
        optimized = reference

    # require an explicit backend when selection remains ambiguous.
    names = ", ".join(sorted(spec.backend for spec in optimized))
    raise KernelSelectionError(
        f"operation {op!r} has multiple eligible backends ({names}) on "
        f"{platform.device_type} {platform.arch}; pass backend=... to choose one"
    )


@compile_safe_cache
def _resolve(
    op: str, backend: str | None, device_type: str, device_index: int | None
) -> Callable[..., Any]:
    return select_kernel(op, backend, _platform_for(device_type, device_index)).fn


def clear_cache() -> None:
    """Drop the platform and resolution caches. Used by tests."""
    _platform_for.cache_clear()
    _resolve.cache_clear()


def dispatch(
    op: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    backend: str | None = None,
    *,
    device: torch.device,
) -> Any:
    return _resolve(op, backend, device.type, device.index)(*args, **kwargs)
