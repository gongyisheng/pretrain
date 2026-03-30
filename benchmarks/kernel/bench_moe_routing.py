import os
import sys
sys.path.insert(0, ".")

import torch
import triton

from src.kernel.triton.moe_routing import triton_moe_routing
from src.kernel.torch.moe_routing import torch_moe_routing


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[1024, 2048, 4096, 8192, 16384],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton (atomic)', 'PyTorch (sort)'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='moe-routing-E64-k2',
        args={'E': 64, 'k': 2, 'capacity_factor': 1.25},
    )
)
def bench_routing_e64(T, E, k, capacity_factor, provider):
    torch.manual_seed(42)
    top_indices = torch.randint(0, E, (T, k), device='cuda')
    top_weights = torch.rand(T, k, device='cuda', dtype=torch.bfloat16)

    fn = triton_moe_routing if provider == 'triton' else torch_moe_routing
    ms = triton.testing.do_bench(lambda: fn(top_indices, top_weights, E, capacity_factor))
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['E'],
        x_vals=[4, 8, 16, 32, 64, 128],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton (atomic)', 'PyTorch (sort)'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='moe-routing-T8192-k2',
        args={'T': 8192, 'k': 2, 'capacity_factor': 1.25},
    )
)
def bench_routing_sweep_experts(T, E, k, capacity_factor, provider):
    torch.manual_seed(42)
    top_indices = torch.randint(0, E, (T, k), device='cuda')
    top_weights = torch.rand(T, k, device='cuda', dtype=torch.bfloat16)

    fn = triton_moe_routing if provider == 'triton' else torch_moe_routing
    ms = triton.testing.do_bench(lambda: fn(top_indices, top_weights, E, capacity_factor))
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['k'],
        x_vals=[1, 2, 4, 8],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton (atomic)', 'PyTorch (sort)'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='moe-routing-T8192-E64',
        args={'T': 8192, 'E': 64, 'capacity_factor': 1.25},
    )
)
def bench_routing_sweep_topk(T, E, k, capacity_factor, provider):
    torch.manual_seed(42)
    top_indices = torch.randint(0, E, (T, k), device='cuda')
    top_weights = torch.rand(T, k, device='cuda', dtype=torch.bfloat16)

    fn = triton_moe_routing if provider == 'triton' else torch_moe_routing
    ms = triton.testing.do_bench(lambda: fn(top_indices, top_weights, E, capacity_factor))
    return ms


if __name__ == '__main__':
    os.makedirs('logs/benchmarks', exist_ok=True)
    bench_routing_e64.run(print_data=True, save_path='logs/benchmarks/')
    bench_routing_sweep_experts.run(print_data=True, save_path='logs/benchmarks/')
    bench_routing_sweep_topk.run(print_data=True, save_path='logs/benchmarks/')
