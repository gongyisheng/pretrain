import os
import sys
sys.path.insert(0, ".")

import torch
import triton

from src.kernel.triton.moe_scatter import (
    triton_moe_scatter_in,
    triton_moe_scatter_out,
    _run_scatter_in,
    _run_scatter_in_bwd,
    _run_scatter_out,
    _run_scatter_out_bwd_expert,
)
from src.kernel.torch.moe_scatter import torch_moe_scatter_in, torch_moe_scatter_out


def _make_routing_data(T, k, E, capacity_factor=1.25):
    """Generate realistic routing data for benchmarks."""
    top_indices = torch.randint(0, E, (T, k), device='cuda')
    flat_expert_ids = top_indices.reshape(-1)
    flat_token_ids = torch.arange(T, device='cuda').unsqueeze(1).expand(T, k).reshape(-1)

    expert_counts = torch.bincount(flat_expert_ids.long(), minlength=E)
    capacity = int(T * k * capacity_factor / E)
    offsets = torch.zeros(E, dtype=torch.long, device='cuda')
    offsets[1:] = expert_counts[:-1].cumsum(0)
    positions = torch.arange(T * k, device='cuda') - offsets[flat_expert_ids.sort(stable=True)[0]]

    sorted_expert_ids, sorted_order = flat_expert_ids.sort(stable=True)
    sorted_token_ids = flat_token_ids[sorted_order]
    keep_mask = positions < capacity
    return (
        sorted_expert_ids[keep_mask],
        sorted_token_ids[keep_mask],
        positions[keep_mask],
        capacity,
    )


# --- Scatter IN benchmarks ---

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[1024, 2048, 4096, 8192, 16384],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='moe-scatter-in-fwd',
        args={'D': 512, 'E': 64, 'k': 2},
    )
)
def bench_scatter_in_fwd(T, D, E, k, provider):
    torch.manual_seed(42)
    x_flat = torch.randn(T, D, device='cuda', dtype=torch.bfloat16)
    se, st, pos, cap = _make_routing_data(T, k, E)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: _run_scatter_in(x_flat, se, st, pos, E, cap))
    else:
        ms = triton.testing.do_bench(lambda: torch_moe_scatter_in(x_flat, se, st, pos, E, cap))
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[1024, 2048, 4096, 8192, 16384],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='moe-scatter-in-bwd',
        args={'D': 512, 'E': 64, 'k': 2},
    )
)
def bench_scatter_in_bwd(T, D, E, k, provider):
    torch.manual_seed(42)
    se, st, pos, cap = _make_routing_data(T, k, E)
    grad_padded = torch.randn(E, cap, D, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: _run_scatter_in_bwd(grad_padded, se, st, pos, T))
    else:
        def torch_bwd():
            x_flat = torch.randn(T, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            padded = torch_moe_scatter_in(x_flat, se, st, pos, E, cap)
            padded.backward(grad_padded)
        ms = triton.testing.do_bench(torch_bwd)
    return ms


# --- Scatter OUT benchmarks ---

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[1024, 2048, 4096, 8192, 16384],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='moe-scatter-out-fwd',
        args={'D': 512, 'E': 64, 'k': 2},
    )
)
def bench_scatter_out_fwd(T, D, E, k, provider):
    torch.manual_seed(42)
    se, st, pos, cap = _make_routing_data(T, k, E)
    expert_out = torch.randn(E, cap, D, device='cuda', dtype=torch.bfloat16)
    weights = torch.rand(se.shape[0], device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: _run_scatter_out(expert_out, se, st, pos, weights, T))
    else:
        ms = triton.testing.do_bench(lambda: torch_moe_scatter_out(expert_out, se, st, pos, weights, T))
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[1024, 2048, 4096, 8192, 16384],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='moe-scatter-out-bwd',
        args={'D': 512, 'E': 64, 'k': 2},
    )
)
def bench_scatter_out_bwd(T, D, E, k, provider):
    torch.manual_seed(42)
    se, st, pos, cap = _make_routing_data(T, k, E)
    weights = torch.rand(se.shape[0], device='cuda', dtype=torch.bfloat16)
    grad_output = torch.randn(T, D, device='cuda', dtype=torch.bfloat16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: _run_scatter_out_bwd_expert(grad_output, se, st, pos, weights, E, cap))
    else:
        def torch_bwd():
            expert_out = torch.randn(E, cap, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            output = torch_moe_scatter_out(expert_out, se, st, pos, weights, T)
            output.backward(grad_output)
        ms = triton.testing.do_bench(torch_bwd)
    return ms


if __name__ == '__main__':
    os.makedirs('logs/benchmarks', exist_ok=True)
    bench_scatter_in_fwd.run(print_data=True, save_path='logs/benchmarks/')
    bench_scatter_in_bwd.run(print_data=True, save_path='logs/benchmarks/')
    bench_scatter_out_fwd.run(print_data=True, save_path='logs/benchmarks/')
    bench_scatter_out_bwd.run(print_data=True, save_path='logs/benchmarks/')
