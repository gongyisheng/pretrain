import os
import sys
sys.path.insert(0, ".")

import torch
import triton

from src.kernel.triton.cross_entropy import triton_cross_entropy, triton_cross_entropy_fwd, triton_cross_entropy_bwd
from src.kernel.torch.cross_entropy import torch_cross_entropy


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'F.cross_entropy'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='cross-entropy-fwd',
        args={'V': 50257},
    )
)
def bench_ce_fwd(M, V, provider):
    logits = torch.randn(M, V, device='cuda', dtype=torch.bfloat16)
    targets = torch.randint(0, V, (M,), device='cuda')

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_cross_entropy_fwd(logits, targets))
    else:
        ms = triton.testing.do_bench(lambda: torch_cross_entropy(logits, targets))

    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[1024, 2048, 4096, 8192, 12288],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'F.cross_entropy'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='cross-entropy-fwd+bwd',
        args={'V': 50257},
    )
)
def bench_ce_fwd_bwd(M, V, provider):
    logits = torch.randn(M, V, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, V, (M,), device='cuda')

    if provider == 'triton':
        def triton_fwd_bwd():
            logits.grad = None
            loss = triton_cross_entropy(logits, targets)
            loss.backward()
        ms = triton.testing.do_bench(triton_fwd_bwd)
    else:
        def torch_fwd_bwd():
            logits.grad = None
            loss = torch_cross_entropy(logits, targets)
            loss.backward()
        ms = triton.testing.do_bench(torch_fwd_bwd)

    return ms


if __name__ == '__main__':
    os.makedirs('logs/benchmarks', exist_ok=True)
    bench_ce_fwd.run(print_data=True, save_path='logs/benchmarks/')
    bench_ce_fwd_bwd.run(print_data=True, save_path='logs/benchmarks/')
