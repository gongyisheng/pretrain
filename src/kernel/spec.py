from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

BuildMode = Literal["eager", "jit", "aot"]


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
    priority: int
    can_implement: CanImplementFn
    build: BuildMode
    autograd: bool

    def __post_init__(self) -> None:
        if type(self.autograd) is not bool:
            raise TypeError(
                f"autograd must be bool, got {type(self.autograd).__name__}"
            )
