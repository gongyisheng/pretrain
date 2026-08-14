from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

BuildMode = Literal["eager", "jit", "aot"]


@dataclass(frozen=True, slots=True)
class SupportResult:
    supported: bool
    reason: str | None = None


AvailabilityFn = Callable[[], SupportResult]
EligibilityFn = Callable[[tuple[Any, ...], Mapping[str, Any]], SupportResult]


@dataclass(frozen=True, slots=True)
class KernelSpec:
    op: str
    backend: str
    fn: Callable[..., Any]
    priority: int
    availability: AvailabilityFn
    eligibility: EligibilityFn
    build: BuildMode
    autograd: bool

    def __post_init__(self) -> None:
        if type(self.autograd) is not bool:
            raise TypeError(
                f"autograd must be bool, got {type(self.autograd).__name__}"
            )
