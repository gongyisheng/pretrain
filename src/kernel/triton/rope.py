import torch
import triton
import triton.language as tl

@triton.jit
def _rope_fwd_kernel(
    X_ptr,
    COS_ptr,
    SIN_ptr,
    Y_ptr,
    seq_stride,
    cos_stride,
    seq,
    d_head,
    BLOCK_SIZE: tl.constexpr,
):
    """RoPE forward: y = x * cos + rotate_half(x) * sin, one row per program.

    Each pair (x[i], x[i+half]) is rotated by angle encoded in (cos, sin).
    rotate_half(x) = cat(-x2, x1) where x1 = x[:half], x2 = x[half:].

    Shape contract:
        X: (B * n_heads * S, d_head) — flattened input
        COS: (S, d_head) — cosine of rotation angles per position
        SIN: (S, d_head) — sine of rotation angles per position
        Y: (B * n_heads * S, d_head) — flattened output

    Args:
        X_ptr: pointer to input tensor
        COS_ptr: pointer to cosine tensor
        SIN_ptr: pointer to sine tensor
        Y_ptr: pointer to output tensor
        seq_stride: elements between consecutive rows in X/Y (= d_head if contiguous)
        cos_stride: elements between consecutive rows in COS/SIN (= d_head if contiguous)
        seq: sequence length S (used to map row_idx to cos/sin row via row_idx % S)
        d_head: head dimension
        BLOCK_SIZE: tile width, must be power of 2 and >= d_head
    """
    row_idx = tl.program_id(0)
    cos_row = row_idx % seq
    off_x = seq_stride * row_idx
    off_cos = cos_stride * cos_row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < d_head
    half = d_head // 2

    x = tl.load(X_ptr + off_x + cols, mask, other=0.0)
    cos = tl.load(COS_ptr + off_cos + cols, mask, other=0.0)
    sin = tl.load(SIN_ptr + off_cos + cols, mask, other=0.0)

    # rotate: y = cat(x1,x2)*cos + cat(-x2,x1)*sin
    first_half = cols < half
    partner_idx = tl.where(first_half, cols + half, cols - half)
    partner = tl.load(X_ptr + off_x + partner_idx, mask=mask)
    rotated = tl.where(first_half, -partner, partner)

    y = x * cos + rotated * sin
    tl.store(Y_ptr + off_x + cols, y, mask=mask)


def triton_rope_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
):
    x = x.contiguous()
    batch, n_heads, seq, d_head = x.shape

    # Flatten x to 2D
    x_flat = x.view(-1, d_head)

    # Flatten cos, sin to 2D
    cos_flat = cos.view(seq, d_head).contiguous()
    sin_flat = sin.view(seq, d_head).contiguous()

    y_flat = torch.empty_like(x_flat)
    num_rows = x_flat.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(d_head)
    grid = (num_rows,)
    _rope_fwd_kernel[grid](
        x_flat, cos_flat, sin_flat, y_flat,
        x_flat.stride(0), cos_flat.stride(0),
        seq, d_head, BLOCK_SIZE
    )
    return y_flat.view(batch, n_heads, seq, d_head)


@triton.jit
def _rope_bwd_kernel(
    DY_ptr,
    COS_ptr,
    SIN_ptr,
    DX_ptr,
    dy_stride,
    cos_stride,
    seq,
    d_head,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cos_row = row_idx % seq
    off_dy = dy_stride * row_idx
    off_cos = cos_stride * cos_row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < d_head
    half = d_head // 2

    dy = tl.load(DY_ptr + off_dy + cols, mask, other=0.0)
    cos = tl.load(COS_ptr + off_cos + cols, mask, other=0.0)
    sin = tl.load(SIN_ptr + off_cos + cols, mask, other=0.0)

    # rotate: y = cat(x1,x2)*cos + cat(-x2,x1)*sin
    first_half = cols < half
    partner_idx = tl.where(first_half, cols + half, cols - half)
    partner = tl.load(DY_ptr + off_dy + partner_idx, mask=mask)
    rotated = tl.where(first_half, -partner, partner)

    dx = dy * cos + rotated * (-sin)
    tl.store(DX_ptr + off_dy + cols, dx, mask)


def triton_rope_bwd(
    dy: torch.Tensor,
    cos: torch.Tensor,
    sin:torch.Tensor
):
    dy = dy.contiguous()
    batch, n_heads, seq, d_head = dy.shape

    dy_flat = dy.view(-1, d_head)
    cos_flat = cos.view(seq, d_head).contiguous()
    sin_flat = sin.view(seq, d_head).contiguous()

    dx_flat = torch.empty_like(dy_flat)
    num_rows = dy_flat.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(d_head)
    grid = (num_rows,)
    _rope_bwd_kernel[grid](
        dy_flat, cos_flat, sin_flat, dx_flat,
        dy_flat.stride(0), cos_flat.stride(0),
        seq, d_head, BLOCK_SIZE
    )

    return dx_flat.view(batch, n_heads, seq, d_head)


class TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return triton_rope_fwd(x, cos, sin)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        return triton_rope_bwd(dy, cos, sin), None, None

def triton_rope(x, cos, sin):
    return TritonRoPE.apply(x, cos, sin)
