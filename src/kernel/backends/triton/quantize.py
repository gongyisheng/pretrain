"""Triton helpers for Blackwell FP4 quantization."""

import torch
import triton
import triton.language as tl
from torch._library.triton import triton_op, wrap_triton

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda


def has_e2m1_rne(device: torch.device) -> bool:
    """Whether `device` supports the Blackwell E2M1 RNE conversion instruction."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] in {
        10,
        11,
        12,
    }


@triton.jit
def _pack_e2m1_rne_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    low = tl.load(x_ptr + 2 * offsets, mask=mask, other=0.0).to(tl.float32)
    high = tl.load(x_ptr + 2 * offsets + 1, mask=mask, other=0.0).to(tl.float32)
    packed = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 fp4;
            cvt.rn.satfinite.e2m1x2.f32 fp4, $1, $2;
            mov.b32 $0, {fp4, 0, 0, 0};
        }
        """,
        constraints="=r,f,f",
        args=[high, low],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    tl.store(out_ptr + offsets, packed.to(tl.uint8), mask=mask)


@register_kernel(
    op="quantize.pack_e2m1_rne",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0), max_arch=(12, 99))}),
)
@triton_op("jit_kernel::pack_e2m1_rne", mutates_args={})
def pack_e2m1_rne(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Convert scaled values into low-nibble-first packed E2M1 codes with RNE."""
    if dim not in (-2, -1):
        raise ValueError(f"dim must be -2 or -1, got {dim}")
    if not has_e2m1_rne(x.device):
        raise RuntimeError("E2M1 hardware RNE requires a Blackwell CUDA device")
    axis = dim % x.ndim
    if x.shape[axis] % 2:
        raise ValueError(
            f"fp4_e2m1 contraction extent must be even, got {x.shape[axis]}"
        )

    x = x.movedim(axis, -1).contiguous()
    out = torch.empty(
        (*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=torch.uint8
    )
    n_elements = out.numel()
    if n_elements:
        wrap_triton(_pack_e2m1_rne_kernel)[
            lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
        ](
            x,
            out,
            n_elements,
            BLOCK=256,
        )
    return out.movedim(-1, axis).contiguous()
