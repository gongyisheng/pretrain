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
        """Rotate along a GEMM contraction dimension, storing as `out_dtype`."""

    @abstractmethod
    def inverse(
        self, x: torch.Tensor, contract_dim: int, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Undo the rotation along a GEMM contraction dimension."""

    @property
    def alignment(self) -> int:
        """Rows a ragged contraction must be padded to before this rotation cancels.

        1 for an elementwise transform, which needs no alignment at all.
        """
        return 1


class HadamardRotation(Rotation):
    def __init__(
        self,
        block_size: int = 16,
        random_sign: bool = True,
        seed: int = 0,
        sign_vector: torch.Tensor | None = None,
    ):
        super().__init__()

        if (
            type(block_size) is not int
            or block_size < 1
            or block_size & (block_size - 1)
        ):
            raise ValueError(
                f"block_size must be a positive power of two, got {block_size!r}"
            )
        if type(random_sign) is not bool:
            raise ValueError(f"random_sign must be a bool, got {random_sign!r}")
        if type(seed) is not int or seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        if sign_vector is not None:
            try:
                signs = torch.as_tensor(sign_vector, dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(
                    f"sign_vector must contain {block_size} values drawn from -1 and 1"
                ) from error
            if (
                tuple(signs.shape) != (block_size,)
                or not torch.all((signs == -1) | (signs == 1)).item()
            ):
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
        if self.block_size == 1:
            return x if out_dtype is None else x.to(out_dtype)
        if contract_dim == -1:
            return rotate(x, self.block_size, self._signs(x), inverse, out_dtype)
        return rotate(
            x.transpose(-2, -1), self.block_size, self._signs(x), inverse, out_dtype
        ).transpose(-2, -1)

    def _signs(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.sign_vector is None:
            return None
        return self.sign_vector.to(device=x.device, dtype=torch.float32)


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


def build_rotation(rotation: dict | None) -> Rotation | None:
    if rotation is None:
        return None
    try:
        rotation_cls = rotation["rotation_cls"]
        rotation_kwargs = rotation["rotation_kwargs"]
        return ROTATION_REGISTRY[rotation_cls](**rotation_kwargs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid quant rotation: {error}") from error
