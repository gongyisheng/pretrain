from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry
from src.kernel.spec import BACKEND_PRIORITIES, CheckResult, KernelSpec


class KernelSelectionError(RuntimeError):
    pass


_SELECTION_CACHE_MAX_ENTRIES = 256
_selection_cache: OrderedDict[tuple[Any, ...], KernelSpec] = OrderedDict()


def clear_selection_cache() -> None:
    _selection_cache.clear()


def _value_cache_key(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.layout,
            value.device,
            value.requires_grad,
        )
    if isinstance(value, torch.dtype):
        return ("dtype", value)
    if type(value) in {bool, bytes, complex, float, int, str, type(None)}:
        return ("scalar", type(value), value)
    return None


def _selection_cache_key(
    op: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Any, ...] | None:
    arg_keys = tuple(_value_cache_key(value) for value in args)
    if any(value is None for value in arg_keys):
        return None
    keyword_keys: list[tuple[str, tuple[Any, ...]]] = []
    for name, value in kwargs.items():
        value_key = _value_cache_key(value)
        if not isinstance(name, str) or value_key is None:
            return None
        keyword_keys.append((name, value_key))
    return op, arg_keys, tuple(sorted(keyword_keys)), torch.is_grad_enabled()


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

    cache_key = None
    if registry is KERNEL_REGISTRY:
        cache_key = _selection_cache_key(op, args, kwargs)
        if cache_key is not None:
            cached = _selection_cache.get(cache_key)
            if cached is not None:
                _selection_cache.move_to_end(cache_key)
                return cached

    rejected: list[str] = []
    for spec in sorted(
        specs,
        key=lambda item: BACKEND_PRIORITIES[item.backend],
        reverse=True,
    ):
        result = _check(spec, args, kwargs)
        if result.ok:
            if cache_key is not None:
                _selection_cache[cache_key] = spec
                _selection_cache.move_to_end(cache_key)
                if len(_selection_cache) > _SELECTION_CACHE_MAX_ENTRIES:
                    _selection_cache.popitem(last=False)
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
