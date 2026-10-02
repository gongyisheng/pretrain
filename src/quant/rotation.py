import hashlib
import inspect
import json
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.kernel.ops.hadamard import rotate


class Rotation(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, x: torch.Tensor, contract_dim: int, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Rotate a matrix axis, storing as `out_dtype`."""

    @abstractmethod
    def inverse(
        self, x: torch.Tensor, contract_dim: int, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Undo the rotation along a matrix axis."""

    @property
    def alignment(self) -> int:
        """Required axis alignment; elementwise transforms need only one element."""
        return 1


def apply_rotation_on_axes(
    tensor: torch.Tensor,
    rotation: Rotation | None,
    axes: tuple[int, ...] | list[int],
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply a rotation to the given matrix axes."""
    out_dtype = tensor.dtype if out_dtype is None else out_dtype
    if rotation is not None:
        for axis in axes:
            tensor = rotation(tensor, axis, out_dtype)
    return tensor.to(out_dtype)


def transpose_rotation_axes(axes: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Swap rotation axes for a matrix transpose."""
    return tuple(-3 - axis for axis in axes)


class HadamardRotation(Rotation):
    def __init__(
        self,
        block_size: int = 16,
        random_sign: bool = True,
        seed: int = 42,
        sign_vector: torch.Tensor | None = None,
    ):
        super().__init__()

        if type(random_sign) is not bool:
            raise ValueError(f"random_sign must be a bool, got {random_sign!r}")
        if type(seed) is not int or seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        if sign_vector is not None:
            signs = torch.as_tensor(sign_vector, dtype=torch.float32)
            if not torch.all((signs == -1) | (signs == 1)).item():
                raise ValueError(
                    f"sign_vector must contain {block_size} values drawn from -1 and 1"
                )
            signs = signs.clone()
        elif random_sign:
            # Pin draws to CPU for reproducible signs across devices.
            generator = torch.Generator().manual_seed(seed)
            draws = torch.rand(block_size, generator=generator, device="cpu")
            signs = torch.where(draws < 0.5, -1.0, 1.0)
        else:
            signs = None

        self.block_size = block_size
        self.random_sign = random_sign
        self.seed = seed
        self.register_buffer("sign_vector", signs)

    @property
    def alignment(self) -> int:
        return self.block_size

    def forward(
        self, x: torch.Tensor, contract_dim: int, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return self._transform(x, contract_dim, False, out_dtype)

    def inverse(
        self, x: torch.Tensor, contract_dim: int, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return self._transform(x, contract_dim, True, out_dtype)

    def _transform(
        self,
        x: torch.Tensor,
        contract_dim: int,
        inverse: bool,
        out_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if contract_dim not in (-2, -1):
            raise ValueError(f"contract_dim must be -2 or -1, got {contract_dim}")
        device = x.device
        if contract_dim == -1:
            return rotate(x, self.block_size, self._signs(device), inverse, out_dtype)
        elif contract_dim == -2:
            return rotate(
                x.transpose(-2, -1),
                self.block_size,
                self._signs(device),
                inverse,
                out_dtype,
            ).transpose(-2, -1)

    def _signs(self, device: torch.device) -> torch.Tensor | None:
        if self.sign_vector is None:
            return None
        return self.sign_vector.to(device=device, dtype=torch.float32)


ROTATION_REGISTRY: dict[str, type[Rotation]] = {"hadamard": HadamardRotation}


def build_rotation_key(
    rotation: dict,
    include: list[str],
    exclude: list[str],
) -> str:
    """Build a stable key from canonical rotation config and module scope."""
    rotation_cls = rotation["rotation_cls"]
    # Normalize defaults so omitted and explicit defaults share one identity.
    signature = inspect.signature(ROTATION_REGISTRY[rotation_cls])
    kwargs = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    kwargs.update(rotation["rotation_kwargs"])
    identity = json.dumps(
        {
            "rotation_cls": rotation_cls,
            "rotation_kwargs": kwargs,
            "include": sorted(include),
            "exclude": sorted(exclude),
        },
        sort_keys=True,
    )
    # Keep 64 digest bits for compact, readable state_dict paths.
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return f"{rotation_cls}-{digest}"


def build_rotation(rotation_cfg: dict | None, seed: int = 42) -> Rotation | None:
    if rotation_cfg is None or rotation_cfg["rotation_cls"] is None:
        return None
    try:
        rotation_kwargs = dict(rotation_cfg["rotation_kwargs"])
        rotation_kwargs["seed"] = seed
        return ROTATION_REGISTRY[rotation_cfg["rotation_cls"]](**rotation_kwargs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid quant rotation: {error}") from error
