from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

BuildMode = Literal["eager", "jit", "aot"]

BACKEND_PRIORITIES = {
    "eager": 0,
    "torch": 1,
    "triton": 2,
    "cublaslt": 3,
}


@dataclass(frozen=True, slots=True)
class CheckResult:
    ok: bool
    reason: str | None = None


CanImplementFn = Callable[[tuple[Any, ...], Mapping[str, Any]], CheckResult]


@dataclass(frozen=True, slots=True)
class KernelSpec:
    op: str
    backend: str
    fn: Callable[..., Any]
    build: BuildMode
    autograd: bool

    def __post_init__(self) -> None:
        if type(self.autograd) is not bool:
            raise TypeError(
                f"autograd must be bool, got {type(self.autograd).__name__}"
            )
        if self.backend not in BACKEND_PRIORITIES:
            supported = ", ".join(BACKEND_PRIORITIES)
            raise ValueError(
                f"unknown kernel backend {self.backend!r}; supported: {supported}"
            )


@dataclass(frozen=True, slots=True)
class OperationSpec:
    can_implement: Mapping[str, CanImplementFn]
    _callbacks: dict[str, CanImplementFn] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        callbacks = dict(self.can_implement)
        if not callbacks:
            raise ValueError("can_implement must be nonempty")
        for backend, callback in callbacks.items():
            if backend not in BACKEND_PRIORITIES:
                supported = ", ".join(BACKEND_PRIORITIES)
                raise ValueError(
                    f"unknown kernel backend {backend!r}; supported: {supported}"
                )
            if not callable(callback):
                raise TypeError(
                    f"backend can_implement must be callable: backend={backend!r}"
                )
        object.__setattr__(self, "_callbacks", callbacks)
        object.__setattr__(self, "can_implement", MappingProxyType(callbacks))

    def supports(self, backend: str) -> bool:
        return backend in self._callbacks

    def validator(self, backend: str) -> CanImplementFn | None:
        return self._callbacks.get(backend)
