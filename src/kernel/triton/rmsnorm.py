import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    A pre-row kernel for RMSNorm forward: y = (x / rms) * weight

    Shape contract:
        X: (M, N) — input, M rows of N elements
        W: (N,)   — learnable scale weight
        Y: (M, N) — output, same shape as X

    Args:
        X_ptr: pointer to input tensor
        W_ptr: pointer to weight vector
        Y_ptr: pointer to output tensor
        stride: number of elements between consecutive rows in X/Y (= N if contiguous)
        N: hidden dimension (number of columns per row)
        eps: epsilon for numerical stability in rsqrt
        BLOCK_SIZE: tile width, must be power of 2 and >= N
    """
    row_idx = tl.program_id(0)
    base_ptr = X_ptr + row_idx * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(base_ptr + cols, mask=mask, other=0.0)
    dtype = x.dtype
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x) / N
    rrms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + cols, mask=mask)
    y = (x * rrms * w).to(dtype)
    tl.store(Y_ptr + row_idx * stride + cols, y, mask=mask)


def triton_rmsnorm_fwd(x, weight, eps=1e-6):
    """
    Launch function of rmsnorm fwd kernel
    """
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _rmsnorm_fwd_kernel[grid](x, weight, y, x.stride(0), N, eps, BLOCK_SIZE)
    return y


@triton.jit
def _rmsnorm_bwd_kernel(
    DY_ptr,
    X_ptr,
    W_ptr,
    DX_ptr,
    DW_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm backward: compute dx and dw from upstream gradient dy, one row per program.

    Shape contract:
        DY: (M, N) — upstream gradient
        X:  (M, N) — input saved from forward
        W:  (N,)   — weight saved from forward
        DX: (M, N) — gradient w.r.t. input
        DW: (N,)   — gradient w.r.t. weight (accumulated via atomic_add across all rows)

    Args:
        DY_ptr: pointer to upstream gradient tensor
        X_ptr: pointer to input tensor (saved from forward)
        W_ptr: pointer to weight vector
        DX_ptr: pointer to output gradient w.r.t. input
        DW_ptr: pointer to output gradient w.r.t. weight (must be zeroed before launch)
        stride: number of elements between consecutive rows (= N if contiguous)
        N: hidden dimension (number of columns per row)
        eps: epsilon for numerical stability in rsqrt
        BLOCK_SIZE: tile width, must be power of 2 and >= N
    """
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = row_idx * stride + cols

    dy = tl.load(DY_ptr + offset, mask=mask, other=0.0)
    dtype = dy.dtype
    dy = dy.to(tl.float32)
    x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x) / N
    rrms = tl.rsqrt(mean_sq + eps)
    x_hat = x * rrms

    dy_w = dy * w
    dx = rrms * (dy_w - x_hat * tl.sum(dy_w * x_hat) / N)

    tl.store(DX_ptr + offset, dx.to(dtype), mask=mask)

    dw = dy * x_hat
    tl.atomic_add(DW_ptr + cols, dw, mask=mask)


def triton_rmsnorm_bwd(dy, x, weight, eps=1e-6):                                                                                                                    
    M, N = x.shape                                                               
    dx = torch.empty_like(x)                                                                                                                                        
    dw = torch.zeros_like(weight, dtype=torch.float32)  # fp32 for atomic_add accumulation
    BLOCK_SIZE = triton.next_power_of_2(N)                                                                                                                          
    grid = (M,)                                                                                                                                                     
    _rmsnorm_bwd_kernel[grid](dy, x, weight, dx, dw, x.stride(0), N, eps, BLOCK_SIZE)
    return dx, dw.to(weight.dtype)


class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return triton_rmsnorm_fwd(x, weight, eps)
    
    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx, dw = triton_rmsnorm_bwd(dy, x, weight, ctx.eps)
        return dx, dw, None


def triton_rmsnorm(x, weight, eps=1e-6):
    return TritonRMSNorm.apply(x, weight, eps)
