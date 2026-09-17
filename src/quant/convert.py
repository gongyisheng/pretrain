import torch.nn as nn

from src.layers.mlp import SparseMoEBlock
from src.quant.constants import QUANT_PASSTHROUGH
from src.quant.linear import QuantizedLinear
from src.quant.moe import QuantizedSparseMoEBlock
from src.quant.rotation import build_rotation, build_rotation_key
from src.quant.utils import should_quantize


def enable_quantization(model: nn.Module) -> None:
    """Enable all quantized modules in the model."""
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedSparseMoEBlock)):
            module.quantization_enabled = True


def apply_quantization(model: nn.Module, config) -> nn.Module:
    """Convert eligible modules before optimizer construction."""
    quantization_config = config.training.quantization
    if not quantization_config.enabled or all(
        fmt in QUANT_PASSTHROUGH
        for per_gemm in quantization_config.dtype.values()
        for fmt in per_gemm.values()
    ):
        return model

    rotation = None
    if quantization_config.rotation is not None:
        key = build_rotation_key(
            quantization_config.rotation,
            quantization_config.include,
            quantization_config.exclude,
        )
        rotation = build_rotation(quantization_config.rotation)
        model.quant_rotations = nn.ModuleDict({key: rotation})
        print(f"quant: rotation {key} <- {quantization_config.rotation}")

    embedding_weight_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
    }

    # nn.Linear swaps to QuantizedLinear; SparseMoEBlock (whose routed experts are
    # stacked Parameters, unreachable as Linears) retypes to QuantizedSparseMoEBlock.
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, (nn.Linear, SparseMoEBlock)):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not should_quantize(full_name, quantization_config):
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
            quantized_module = quantized_cls.from_module(
                child, quantization_config, rotation=rotation
            )
            setattr(parent, child_name, quantized_module)
    return model
