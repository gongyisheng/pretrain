import pytest
import torch

from benchmarks.gemm.bench_grouped_gemm import _grouped_gemm_with_autograd


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped benchmark needs CUDA"
)
@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_grouped_gemm_benchmark_backward_path_runs(backend):
    torch.manual_seed(0)
    a = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(2, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    offs = torch.tensor([16, 32], device="cuda", dtype=torch.int32)

    out = _grouped_gemm_with_autograd(a, b, offs, bias, backend)
    out.backward(torch.randn_like(out))

    assert out.shape == (32, 16)
    for tensor in (a, b, bias):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
