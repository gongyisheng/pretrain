import os
import sys
sys.path.insert(0, ".")

import torch
import triton

from src.kernel.triton.rope import triton_rope_fwd, triton_rope_bwd
from src.kernel.torch.rope import torch_rope


def _make_inputs(B, n_heads, S, d_head, dtype=torch.bfloat16):
    x = torch.randn(B, n_heads, S, d_head, device='cuda', dtype=dtype)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device='cuda').float() / d_head))
    positions = torch.arange(S, device='cuda').float()
    angles = positions[:, None] * freqs[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    cos = torch.cos(angles)[None, None, :, :].to(dtype)
    sin = torch.sin(angles)[None, None, :, :].to(dtype)
    return x, cos, sin


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['d_head'],
        x_vals=[64, 96, 128, 192, 256],
        line_arg='provider',
        line_vals=['triton', 'torch_compile'],
        line_names=['Triton', 'torch.compile'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='rope-fwd',
        args={'B': 8, 'n_heads': 12, 'S': 512},
    )
)
def bench_rope_fwd(B, n_heads, S, d_head, provider):
    x, cos, sin = _make_inputs(B, n_heads, S, d_head)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_rope_fwd(x, cos, sin))
    else:
        ms = triton.testing.do_bench(lambda: torch_rope(x, cos, sin))

    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['d_head'],
        x_vals=[64, 96, 128, 192, 256],
        line_arg='provider',
        line_vals=['triton', 'torch_compile'],
        line_names=['Triton', 'torch.compile'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='rope-bwd',
        args={'B': 8, 'n_heads': 12, 'S': 512},
    )
)
def bench_rope_bwd(B, n_heads, S, d_head, provider):
    x, cos, sin = _make_inputs(B, n_heads, S, d_head)
    x.requires_grad_(True)
    dy = torch.randn_like(x)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_rope_bwd(dy, cos, sin))
    else:
        def torch_bwd():
            x.grad = None
            y = torch_rope(x, cos, sin)
            y.backward(dy)
        ms = triton.testing.do_bench(torch_bwd)

    return ms


if __name__ == '__main__':
    os.makedirs('logs/benchmarks', exist_ok=True)
    bench_rope_fwd.run(print_data=True, save_path='logs/benchmarks/')
    bench_rope_bwd.run(print_data=True, save_path='logs/benchmarks/')
