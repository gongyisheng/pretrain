import pytest
import torch

from benchmarks.gemm import bench_grouped_gemm

_grouped_gemm_with_autograd = bench_grouped_gemm._grouped_gemm_with_autograd


def test_grouped_gemm_benchmark_relative_error():
    eager = torch.tensor([1.0, 2.0], device="cpu")
    optimized = torch.tensor([1.0, 2.5], device="cpu")

    assert hasattr(bench_grouped_gemm, "_relative_error")
    assert bench_grouped_gemm._relative_error(optimized, eager) == pytest.approx(
        0.5 / 5**0.5
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped benchmark needs CUDA"
)
@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_grouped_gemm_benchmark_backward_path_runs(backend):
    torch.manual_seed(0)
    a = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(2, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = None
    if backend == "triton":
        bias = torch.randn(
            2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
    offs = torch.tensor([16, 32], device="cuda", dtype=torch.int32)

    out = _grouped_gemm_with_autograd(a, b, offs, bias, backend)
    out.backward(torch.randn_like(out))

    assert out.shape == (32, 16)
    tensors = (a, b) if bias is None else (a, b, bias)
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
