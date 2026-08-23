from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class Rotation(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, contract_dim: int) -> torch.Tensor:
        """Apply the rotation along a GEMM contraction dimension."""

    @abstractmethod
    def inverse(self, x: torch.Tensor, contract_dim: int) -> torch.Tensor:
        """Apply the inverse rotation along a GEMM contraction dimension."""


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
            # CPU-pinned: CUDA's Philox and CPU's MT19937 draw differently from the
            # same seed, so a default-device generator would tie the signs to
            # wherever the model happened to be built.
            generator = torch.Generator().manual_seed(seed)
            draws = torch.rand(block_size, generator=generator, device="cpu")
            signs = torch.where(draws < 0.5, -1.0, 1.0)
        else:
            signs = None

        self.block_size = block_size
        self.random_sign = random_sign
        self.seed = seed
        self.register_buffer("sign_vector", signs)

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
        if contract_dim == -2:
            return self._transform(x.mT, -1, inverse).mT.contiguous()

        blocks = x.float().reshape(*x.shape[:-1], -1, self.block_size)
        blocks = blocks * (1 / math.sqrt(self.block_size))
        signs = self.sign_vector
        if signs is not None:
            signs = signs.to(x.device)
            if not inverse:
                blocks = blocks * signs
        stride = 1
        while stride < self.block_size:
            pairs = blocks.reshape(*blocks.shape[:-1], -1, 2, stride)
            left, right = pairs.unbind(-2)
            blocks = torch.stack((left + right, left - right), dim=-2).reshape(
                blocks.shape
            )
            stride *= 2
        if signs is not None and inverse:
            blocks = blocks * signs
        return blocks.reshape(x.shape).to(x.dtype)


ROTATION_REGISTRY: dict[str, type[Rotation]] = {"hadamard": HadamardRotation}


def build_rotation(rotation: Mapping[str, Any] | None) -> Rotation | None:
    if rotation is None:
        return None
    try:
        rotation_cls = rotation["rotation_cls"]
        rotation_kwargs = rotation["rotation_kwargs"]
        return ROTATION_REGISTRY[rotation_cls](**rotation_kwargs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid quant rotation: {error}") from error
