from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Callable

import torch

from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry
from src.kernel.spec import KernelSpec, PlatformInfo


class KernelSelectionError(RuntimeError):
    pass


@lru_cache(maxsize=None)
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
    if len(specs) == 1:
        return specs[0]
    if platform is None:
        raise KernelSelectionError(
            f"operation {op!r} has multiple backends ({registered}) and no "
            "platform was supplied to choose among them"
        )
    eligible = [spec for spec in specs if spec.is_available(platform)]
    if len(eligible) == 1:
        return eligible[0]
    if not eligible:
        raise KernelSelectionError(
            f"operation {op!r} has no backend for {platform.device_type} "
            f"{platform.arch}; registered: {registered}"
        )
    names = ", ".join(sorted(spec.backend for spec in eligible))
    raise KernelSelectionError(
        f"operation {op!r} has multiple eligible backends ({names}) on "
        f"{platform.device_type} {platform.arch}; pass backend=... to choose one"
    )


@lru_cache(maxsize=None)
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
