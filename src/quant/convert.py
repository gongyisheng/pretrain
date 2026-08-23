from __future__ import annotations

import torch.nn as nn

from src.layers.mlp import SparseMoEBlock
from src.quant.constants import QUANT_PASSTHROUGH
from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock
from src.quant.rotation import build_rotation, build_rotation_key
from src.quant.utils import resolve_quantization_config


def _is_passthrough(quantization_config) -> bool:
    return quantization_config is None or all(
        fmt in QUANT_PASSTHROUGH
        for per_gemm in quantization_config.dtype.values()
        for fmt in per_gemm.values()
    )


def apply_quantization(model: nn.Module, config) -> nn.Module:
    """Swap eligible nn.Linear / SparseMoEBlock modules to their quantized
    counterparts per the run's quant configurations.
    """
    quantization_configs = config.training.quantization
    if not any(qc.enabled for qc in quantization_configs):
        return model

    rotations = {}
    for quantization_config in quantization_configs:
        if not quantization_config.enabled or quantization_config.rotation is None:
            continue
        key = build_rotation_key(
            quantization_config.rotation,
            quantization_config.include,
            quantization_config.exclude,
        )
        # Rules that agree on both the transform and the scope share one Rotation.
        if key not in rotations:
            rotations[key] = build_rotation(quantization_config.rotation)
            print(f"quant: rotation {key} <- {quantization_config.rotation}")
    if rotations:
        model.quant_rotations = nn.ModuleDict(rotations)

    embedding_weight_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)
    }

    # nn.Linear swaps to QuantizedLinear; SparseMoEBlock (whose routed experts are
    # stacked Parameters, unreachable as Linears) retypes to QuantizedSparseMoEBlock.
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, (nn.Linear, SparseMoEBlock)):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            quantization_config = resolve_quantization_config(
                full_name, quantization_configs
            )
            # skip configs that leave every operand in a passthrough dtype
            if _is_passthrough(quantization_config):
                continue
            if isinstance(child, nn.Linear):
                if id(child.weight) in embedding_weight_ids:
                    print(
                        f"quant: skipping {full_name!r} — its weight is tied to an "
                        "embedding; swapping would break the tie."
                    )
                    continue
                quantized_cls = QuantizedLinear
            else:
                quantized_cls = QuantizedSparseMoEBlock
            rotation = None
            if quantization_config.rotation is not None:
                rotation = rotations[
                    build_rotation_key(
                        quantization_config.rotation,
                        quantization_config.include,
                        quantization_config.exclude,
                    )
                ]
            qmod = quantized_cls.from_module(
                child, quantization_config, rotation=rotation
            )
            setattr(parent, child_name, qmod)
    return model
