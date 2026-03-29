import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(
    GATE_ptr,
    UP_ptr,
    Y_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU forward: y = silu(gate) * up, flat element-wise fusion.

    Shape contract:
        GATE: (M * N,) — flattened gate tensor
        UP:   (M * N,) — flattened up tensor
        Y:    (M * N,) — flattened output tensor

    Args:
        GATE_ptr: pointer to gate tensor (flattened)
        UP_ptr: pointer to up tensor (flattened)
        Y_ptr: pointer to output tensor (flattened)
        num_elements: total number of elements (M * N)
        BLOCK_SIZE: number of elements per program
    """
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < num_elements

    gate = tl.load(GATE_ptr + off, mask, other=0.0)
    gate_f32 = gate.to(tl.float32)
    up = tl.load(UP_ptr + off, mask, other=0.0)

    y = gate * tl.sigmoid(gate_f32).to(gate.dtype) * up
    tl.store(Y_ptr + off, y, mask=mask)


def triton_swiglu_fwd(
    gate: torch.Tensor,
    up: torch.Tensor
):
    assert gate.shape == up.shape
    gate = gate.contiguous()
    up = up.contiguous()

    y = torch.empty_like(gate)
    num_elements = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    _swiglu_fwd_kernel[grid](gate, up, y, num_elements, BLOCK_SIZE)
    return y


@triton.jit
def _swiglu_bwd_kernel(
    DY_ptr,
    GATE_ptr,
    UP_ptr,
    DGATE_ptr,
    DUP_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    off = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = off < num_elements

    dy = tl.load(DY_ptr + off, mask=mask, other=0.0)
    gate = tl.load(GATE_ptr + off, mask=mask, other=0.0)
    gate_f32 = gate.to(tl.float32)
    up = tl.load(UP_ptr + off, mask=mask, other=0.0)
    sig = tl.sigmoid(gate_f32).to(gate.dtype)
    dup = dy * gate * sig
    dgate = dy * up * sig * (1+gate*(1-sig))
    tl.store(DGATE_ptr + off, dgate, mask=mask)
    tl.store(DUP_ptr + off, dup, mask=mask)


def triton_swiglu_bwd(
    dy: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor
):
    dy = dy.contiguous()
    gate = gate.contiguous()
    up = up.contiguous()

    dgate = torch.empty_like(gate)
    dup = torch.empty_like(up)
    num_elements = gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    _swiglu_bwd_kernel[grid](dy, gate, up, dgate, dup, num_elements, BLOCK_SIZE)
    return dgate, dup


class TritonSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return triton_swiglu_fwd(gate, up)

    @staticmethod
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        dgate, dup = triton_swiglu_bwd(dy, gate, up)
        return dgate, dup


def triton_swiglu(gate, up):
    return TritonSwiGLU.apply(gate, up)
