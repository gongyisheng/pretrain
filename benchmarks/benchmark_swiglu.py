import os
import sys
sys.path.insert(0, ".")

import torch
import triton

from src.kernel.triton.swiglu import triton_swiglu_fwd, triton_swiglu_bwd
from src.kernel.torch.swiglu import torch_swiglu


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512, 768, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch_compile'],
        line_names=['Triton', 'torch.compile'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='swiglu-fwd',
        args={'M': 4096},
    )
)
def bench_swiglu_fwd(M, N, provider):
    gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_swiglu_fwd(gate, up))
    else:
        ms = triton.testing.do_bench(lambda: torch_swiglu(gate, up))

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
        plot_name='swiglu-bwd',
        args={'M': 4096},
    )
)
def bench_swiglu_bwd(M, N, provider):
    gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_swiglu_bwd(dy, gate, up))
    else:
        def torch_bwd():
            gate.grad = None
            up.grad = None
            y = torch_swiglu(gate, up)
            y.backward(dy)
        ms = triton.testing.do_bench(torch_bwd)

    return ms


if __name__ == '__main__':
    os.makedirs('logs/benchmarks', exist_ok=True)
    bench_swiglu_fwd.run(print_data=True, save_path='logs/benchmarks/')
    bench_swiglu_bwd.run(print_data=True, save_path='logs/benchmarks/')
