import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    Y_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    off = row_idx * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + off + cols, mask, other=0.0)
    dtype = x.dtype
    x = x.to(tl.float32)
    w = tl.load(W_ptr + cols, mask, other=0.0)
    b = tl.load(B_ptr + cols, mask, other=0.0)

    mean = tl.sum(x) / N
    x = x - mean
    var = tl.sum(x * x) / N
    rrms = tl.rsqrt(var + eps)
    y = (x * rrms * w + b).to(dtype)
    tl.store(Y_ptr + off + cols, y, mask)


def triton_layernorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps = 1e-5,
):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _layernorm_fwd_kernel[grid](
        x, weight, bias, y,
        x.stride(0), N, eps, BLOCK_SIZE
    )
    return y


@triton.jit
def _layernorm_bwd_kernel(
    DY_ptr,
    X_ptr,
    W_ptr,
    DX_ptr,
    DW_ptr,
    DB_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    off = row_idx * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(DY_ptr + off + cols, mask, other=0.0)
    dtype = dy.dtype
    dy = dy.to(tl.float32)
    x = tl.load(X_ptr + off + cols, mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask, other=0.0).to(tl.float32)

    mean = tl.sum(x) / N
    x = x - mean
    var = tl.sum(x * x) / N
    rstd = tl.rsqrt(var + eps)
    x_hat = x * rstd
    dy_w = dy * w
    dx = rstd * (dy_w - x_hat * tl.sum(dy_w * x_hat) / N - tl.sum(dy_w) / N)
    tl.store(DX_ptr + off + cols, dx.to(dtype), mask)

    tl.atomic_add(DB_ptr + cols, dy, mask=mask)
    tl.atomic_add(DW_ptr + cols, dy * x_hat, mask=mask)

def triton_layernorm_bwd(
    dy: torch.Tensor, 
    x: torch.Tensor, 
    weight: torch.Tensor, 
    eps = 1e-6
):
    dy = dy.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros(N, dtype=torch.float32, device=x.device)
    db = torch.zeros(N, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _layernorm_bwd_kernel[grid](
        dy, x, weight, dx, dw, db,
        x.stride(0), N, eps, BLOCK_SIZE
    )
    return dx, dw.to(weight.dtype), db.to(weight.dtype)



class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return triton_layernorm_fwd(x, weight, bias, eps)
    
    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx, dw, db = triton_layernorm_bwd(dy, x, weight, ctx.eps)
        return dx, dw, db, None


def triton_layernorm(x, weight, bias, eps=1e-6):
    return TritonLayerNorm.apply(x, weight, bias, eps)
