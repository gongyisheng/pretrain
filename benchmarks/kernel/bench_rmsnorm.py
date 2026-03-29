import os
import sys
sys.path.insert(0, ".")

import torch
import triton

from src.kernel.triton.rmsnorm import triton_rmsnorm_fwd, triton_rmsnorm_bwd
from src.kernel.torch.rmsnorm import torch_rmsnorm

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512, 768, 1024, 2048, 4096, 8192, 16384],
        line_arg='provider',
        line_vals=['triton', 'torch_compile'],
        line_names=['Triton', 'torch.compile'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='rmsnorm-fwd',
        args={'M': 4096},
    )
)
def bench_rmsnorm_fwd(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(N, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_rmsnorm_fwd(x, w))
    else:
        ms = triton.testing.do_bench(lambda: torch_rmsnorm(x, w, 1e-6))

    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512, 768, 1024, 2048, 4096, 8192, 16384],
        line_arg='provider',
        line_vals=['triton', 'torch_compile'],
        line_names=['Triton', 'torch.compile'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='rmsnorm-bwd',
        args={'M': 4096},
    )
)
def bench_rmsnorm_bwd(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_rmsnorm_bwd(dy, x, w))
    else:
        def torch_bwd():
            x.grad = None
            w.grad = None
            y = torch_rmsnorm(x, w, 1e-6)
            y.backward(dy)
        ms = triton.testing.do_bench(torch_bwd)

    return ms


if __name__ == '__main__':
    os.makedirs('logs/benchmarks', exist_ok=True)
    bench_rmsnorm_fwd.run(print_data=True, save_path='logs/benchmarks/')
    bench_rmsnorm_bwd.run(print_data=True, save_path='logs/benchmarks/')
