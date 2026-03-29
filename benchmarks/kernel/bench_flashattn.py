import os
import sys
sys.path.insert(0, ".")

import torch
import triton

from src.kernel.triton.flashattn import triton_flash_attn_fwd, triton_flash_attn_bwd
from src.kernel.torch.flashattn import torch_flash_attn


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'F.scaled_dot_product_attention'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='flashattn-fwd-causal',
        args={'B': 4, 'n_heads': 8, 'd_head': 64},
    )
)
def bench_flashattn_fwd(B, n_heads, seq_len, d_head, provider):
    q = torch.randn(B, n_heads, seq_len, d_head, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, n_heads, seq_len, d_head, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, n_heads, seq_len, d_head, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_flash_attn_fwd(q, k, v, causal=True))
    else:
        ms = triton.testing.do_bench(lambda: torch_flash_attn(q, k, v, causal=True))

    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'F.scaled_dot_product_attention'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='flashattn-bwd-causal',
        args={'B': 4, 'n_heads': 8, 'd_head': 64},
    )
)
def bench_flashattn_bwd(B, n_heads, seq_len, d_head, provider):
    q = torch.randn(B, n_heads, seq_len, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, n_heads, seq_len, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, n_heads, seq_len, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(B, n_heads, seq_len, d_head, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        o, L = triton_flash_attn_fwd(q.detach(), k.detach(), v.detach(), causal=True)
        ms = triton.testing.do_bench(
            lambda: triton_flash_attn_bwd(q.detach(), k.detach(), v.detach(), o, L, do, causal=True)
        )
    else:
        def torch_bwd():
            q.grad = None
            k.grad = None
            v.grad = None
            o = torch_flash_attn(q, k, v, causal=True)
            o.backward(do)
        ms = triton.testing.do_bench(torch_bwd)

    return ms


if __name__ == '__main__':
    os.makedirs('logs/benchmarks', exist_ok=True)
    bench_flashattn_fwd.run(print_data=True, save_path='logs/benchmarks/')
    bench_flashattn_bwd.run(print_data=True, save_path='logs/benchmarks/')
