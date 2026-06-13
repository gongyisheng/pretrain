"""FP8 training support via torchao.float8 module swap.

`convert_to_float8_training` rewrites `nn.Linear` submodules to `Float8Linear`,
which casts inputs and weights to FP8 (E4M3/E5M2) on each forward and dispatches
to cuBLASLt's FP8 GEMM via `torch._scaled_mm`. Activations, biases, and the
surrounding elementwise ops remain in bf16 and are fused by the outer
`torch.compile` pass.

Requirements:
  - GPU with SM 9.0+ (Hopper / Blackwell).
  - Linear shapes with inner/outer dims divisible by 16 (this codebase already
    pads vocab to 128 and uses d_model/intermediate sizes that satisfy this).
  - Call BEFORE `torch.compile(model)` so the swap is visible to the tracer.

MoE expert FFN paths store weights as raw `nn.Parameter` tensors consumed by
`bmm`, not by `nn.Linear`, and are not converted. We disable FP8 for the MoE
architecture rather than silently leaving most of the FLOPs in bf16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.utils.config import TrainConfig

# Recipe names accepted by torchao's Float8LinearConfig.from_recipe_name.
FP8_RECIPES = frozenset({"tensorwise", "rowwise", "rowwise_with_gw_hp"})


def _fp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def maybe_convert_to_fp8(model: nn.Module, config: TrainConfig) -> nn.Module:
    """In-place swap of eligible `nn.Linear` modules to `Float8Linear`.

    Returns the model (possibly mutated). No-op when `training.fp8` is disabled.
    Raises if FP8 is requested but the environment cannot support it — silent
    fallback would hide a config error.
    """
    if not config.training.fp8:
        return model

    if config.model.mlp_cls == "moe":
        raise ValueError(
            "training.fp8=true is not supported for mlp_cls='moe': expert FFN weights "
            "are raw nn.Parameter tensors and are not eligible for the module swap. "
            "Set training.fp8=false or use a dense MLP."
        )

    if not _fp8_supported():
        raise RuntimeError(
            "training.fp8=true requires a CUDA GPU with compute capability >= 9.0 "
            "(Hopper or Blackwell). Disable training.fp8 or run on supported hardware."
        )

    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    recipe = config.training.fp8_recipe
    try:
        fp8_config = Float8LinearConfig.from_recipe_name(recipe)
    except (ValueError, KeyError, AssertionError) as e:
        raise ValueError(
            f"Unknown training.fp8_recipe={recipe!r}. "
            f"Expected a torchao recipe name (e.g. 'tensorwise', 'rowwise')."
        ) from e

    exclude_lm_head = config.training.fp8_exclude_lm_head

    def module_filter_fn(_mod: nn.Module, fqn: str) -> bool:
        if exclude_lm_head and "lm_head" in fqn:
            return False
        # The MoE router gate is a small (d_model, n_experts) projection; FP8 here
        # has no perf win and adds noise to expert assignments.
        if "router" in fqn or "gate" == fqn.rsplit(".", 1)[-1]:
            return False
        return True

    convert_to_float8_training(
        model, config=fp8_config, module_filter_fn=module_filter_fn
    )
    return model
