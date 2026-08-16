from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    can_implement: CanImplementFn
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
