import os
import sys
sys.path.insert(0, ".")

import torch
import triton

from src.kernel.triton.layernorm import triton_layernorm_fwd, triton_layernorm_bwd
from src.kernel.torch.layernorm import torch_layernorm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512, 768, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch_compile'],
        line_names=['Triton', 'torch.compile'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='layernorm-fwd',
        args={'M': 4096},
    )
)
def bench_layernorm_fwd(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(N, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(N, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_layernorm_fwd(x, w, b))
    else:
        ms = triton.testing.do_bench(lambda: torch_layernorm(x, w, b))

    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512, 768, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch_compile'],
        line_names=['Triton', 'torch.compile'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='layernorm-bwd',
        args={'M': 4096},
    )
)
def bench_layernorm_bwd(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_layernorm_bwd(dy, x, w))
    else:
        def torch_bwd():
            x.grad = None
            w.grad = None
            b.grad = None
            y = torch_layernorm(x, w, b)
            y.backward(dy)
        ms = triton.testing.do_bench(torch_bwd)

    return ms


if __name__ == '__main__':
    os.makedirs('logs/benchmarks', exist_ok=True)
    bench_layernorm_fwd.run(print_data=True, save_path='logs/benchmarks/')
    bench_layernorm_bwd.run(print_data=True, save_path='logs/benchmarks/')
