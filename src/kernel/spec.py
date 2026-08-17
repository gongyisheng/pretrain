from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

BuildMode = Literal["eager", "jit", "aot"]

BACKENDS = frozenset({"eager", "triton", "cublaslt"})


@dataclass(frozen=True, slots=True)
class PlatformInfo:
    """A snapshot of the runtime accelerator.

    An absent or invisible accelerator yields the default CPU platform.
    """

    device_type: str = "cpu"
    arch: tuple[int, int] | None = None

    @classmethod
    def detect(cls) -> "PlatformInfo":
        if not torch.cuda.is_available():
            return cls()
        return cls("cuda", torch.cuda.get_device_capability())


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    """One eligibility clause: a device type and an inclusive SM window."""

    device: str
    min_arch: tuple[int, int] | None = None
    max_arch: tuple[int, int] | None = None

    def is_satisfied_by(self, platform: PlatformInfo) -> bool:
        if self.device != platform.device_type:
            return False
        if self.min_arch is None and self.max_arch is None:
            return True
        if platform.arch is None:
            return False
        if self.min_arch is not None and platform.arch < self.min_arch:
            return False
        return self.max_arch is None or platform.arch <= self.max_arch


CPU = CapabilityRequirement("cpu")


def cuda(
    min_arch: tuple[int, int] | None = None,
    max_arch: tuple[int, int] | None = None,
) -> CapabilityRequirement:
    return CapabilityRequirement("cuda", min_arch, max_arch)


@dataclass(frozen=True, slots=True)
class KernelSpec:
    op: str
    backend: str
    fn: Callable[..., Any]
    build: BuildMode
    autograd: bool
    capabilities: frozenset[CapabilityRequirement] = frozenset()

    def __post_init__(self) -> None:
        if type(self.autograd) is not bool:
            raise TypeError(
                f"autograd must be bool, got {type(self.autograd).__name__}"
            )
        if self.backend not in BACKENDS:
            supported = ", ".join(sorted(BACKENDS))
            raise ValueError(
                f"unknown kernel backend {self.backend!r}; supported: {supported}"
            )

    def is_available(self, platform: PlatformInfo) -> bool:
        """Metadata-only check: never inspects tensors."""
        if not self.capabilities:
            return True
        return any(
            requirement.is_satisfied_by(platform) for requirement in self.capabilities
        )
