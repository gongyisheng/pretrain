import hashlib
import inspect
import json
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Rotation(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, contract_dim: int) -> torch.Tensor:
        """Rotate along a GEMM contraction dimension."""

    @abstractmethod
    def inverse(self, x: torch.Tensor, contract_dim: int) -> torch.Tensor:
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

        # Sylvester construction, entries +-1. Kept unscaled so every product in the
        # transform is an exact sign flip; the 1/sqrt(block) normalization is applied
        # to the operand first, which also keeps fp16 sums inside fp32's range.
        hadamard = torch.ones(1, 1)
        while hadamard.shape[0] < block_size:
            hadamard = torch.cat(
                [
                    torch.cat([hadamard, hadamard], 1),  # (N, 2N)
                    torch.cat([hadamard, -hadamard], 1),  # (N, 2N)
                ],
                0,
            )  # (2N, 2N)

        self.block_size = block_size
        self.random_sign = random_sign
        self.seed = seed
        self.register_buffer("sign_vector", signs)
        # Derived from block_size, so it stays out of checkpoints.
        self.register_buffer("hadamard", hadamard, persistent=False)

    @property
    def alignment(self) -> int:
        return self.block_size

    def forward(self, x: torch.Tensor, contract_dim: int) -> torch.Tensor:
        return self._transform(x, contract_dim, inverse=False)

    def inverse(self, x: torch.Tensor, contract_dim: int) -> torch.Tensor:
        return self._transform(x, contract_dim, inverse=True)

    def _transform(
        self, x: torch.Tensor, contract_dim: int, inverse: bool
    ) -> torch.Tensor:
        if contract_dim not in (-2, -1):
            raise ValueError(f"contract_dim must be -2 or -1, got {contract_dim}")
        if self.block_size == 1:
            return x
        length = x.shape[contract_dim]
        if length % self.block_size:
            raise ValueError(
                f"block {self.block_size} must divide the rotated extent {length}"
            )

        hadamard = self.hadamard.to(device=x.device, dtype=torch.float32)
        signs = self.sign_vector
        if signs is not None:
            signs = signs.to(device=x.device, dtype=torch.float32)
        scale = 1 / math.sqrt(self.block_size)
        if contract_dim == -1:
            blocks = x.float().reshape(*x.shape[:-1], -1, self.block_size) * scale
            if signs is not None and not inverse:
                blocks = blocks * signs
            blocks = blocks @ hadamard
            if signs is not None and inverse:
                blocks = blocks * signs
        else:
            blocks = (
                x.float().reshape(*x.shape[:-2], -1, self.block_size, x.shape[-1])
                * scale
            )
            if signs is not None and not inverse:
                blocks = blocks * signs[:, None]
            blocks = hadamard @ blocks
            if signs is not None and inverse:
                blocks = blocks * signs[:, None]
        return blocks.reshape(x.shape).to(x.dtype)


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
