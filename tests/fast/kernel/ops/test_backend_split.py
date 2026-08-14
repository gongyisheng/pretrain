import pytest
import torch

from src.kernel.ops.gemm import grouped_gemm, scaled_gemm, scaled_grouped_gemm
from src.kernel.registry import KERNEL_REGISTRY
from src.kernel.selector import KernelSelectionError


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimized Torch GEMM requires CUDA"
)


@pytest.mark.parametrize("op", ["gemm.grouped", "gemm.scaled", "gemm.scaled_grouped"])
def test_gemm_registry_contains_reference_torch_and_triton(op):
    implementations = KERNEL_REGISTRY.implementations(op)
    backends = {spec.backend for spec in implementations}
    assert backends == {"reference", "torch", "triton"}
    priorities = {spec.backend: spec.priority for spec in implementations}
    assert priorities["triton"] > priorities["torch"] > priorities["reference"]


def test_reference_grouped_gemm_is_a_cpu_oracle_with_gradients():
    a = torch.randn(5, 8, device="cpu", dtype=torch.float32, requires_grad=True)
    b = torch.randn(2, 8, 6, device="cpu", dtype=torch.float32, requires_grad=True)
    offs = torch.tensor([2, 5], device="cpu", dtype=torch.int32)

    out = grouped_gemm(a, b, offs, backend="reference")
    expected = torch.cat((a[:2] @ b[0], a[2:] @ b[1]))

    torch.testing.assert_close(out, expected)
    out.sum().backward()
    assert a.grad is not None
    assert b.grad is not None


def test_forced_torch_grouped_gemm_rejects_cpu_without_fallback():
    a = torch.randn(4, 8, device="cpu")
    b = torch.randn(1, 8, 6, device="cpu")
    offs = torch.tensor([4], device="cpu", dtype=torch.int32)

    with pytest.raises(KernelSelectionError, match="torch.*CUDA"):
        grouped_gemm(a, b, offs, backend="torch")


def test_auto_scaled_gemm_uses_reference_for_cpu_blockwise_inputs():
    aq = torch.tensor([[1, -2, 3, 4], [-1, 2, 0, 3]], device="cpu", dtype=torch.int8)
    bq = torch.tensor(
        [[1, 2, -1], [0, -2, 3], [2, 1, 1], [-1, 0, 2]],
        device="cpu",
        dtype=torch.int8,
    )
    sa = torch.tensor([[0.5, 0.25], [0.75, 0.5]], device="cpu")
    sb = torch.tensor([[1.0, 0.5, 0.25], [0.5, 1.0, 2.0]], device="cpu")

    expected = scaled_gemm(aq, bq, sa, sb, torch.float32, 2, backend="reference")
    actual = scaled_gemm(aq, bq, sa, sb, torch.float32, 2)

    torch.testing.assert_close(actual, expected)


@cuda_only
def test_torch_grouped_gemm_matches_reference_for_supported_inputs():
    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    expected = grouped_gemm(a, b, offs, backend="reference")
    actual = grouped_gemm(a, b, offs, backend="torch")

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@cuda_only
def test_forced_torch_grouped_gemm_rejects_unsupported_bias():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
    bias = torch.randn(2, 48, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(KernelSelectionError, match="torch.*does not support bias"):
        grouped_gemm(a, b, offs, bias=bias, backend="torch")


@cuda_only
def test_torch_scaled_gemm_matches_reference_for_supported_inputs():
    torch.manual_seed(0)
    aq = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
    bq = torch.randn(64, 48, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(64, 1, device="cuda", dtype=torch.float32)
    sb = torch.rand(1, 48, device="cuda", dtype=torch.float32)
    bias = torch.randn(48, device="cuda", dtype=torch.bfloat16)

    expected = scaled_gemm(
        aq, bq, sa, sb, torch.bfloat16, 0, bias=bias, backend="reference"
    )
    actual = scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 0, bias=bias, backend="torch")

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_forced_torch_scaled_gemm_rejects_unsupported_cpu_inputs():
    aq = torch.ones(16, 16, device="cpu", dtype=torch.int8)
    bq = torch.ones(16, 16, device="cpu", dtype=torch.int8)
    sa = torch.ones(16, 1, device="cpu")
    sb = torch.ones(1, 16, device="cpu")

    with pytest.raises(KernelSelectionError, match="torch.*CUDA"):
        scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 0, backend="torch")


@cuda_only
def test_external_torch_grouped_gemm_is_not_selected_for_direct_autograd():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    with pytest.raises(KernelSelectionError, match="autograd mode 'external'"):
        grouped_gemm(a, b, offs, backend="torch")

    out = grouped_gemm(a, b, offs)
    out.float().sum().backward()
    assert a.grad is not None
    assert b.grad is not None


@cuda_only
def test_torch_scaled_grouped_gemm_matches_reference_when_architecture_supports_it():
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("F.scaled_grouped_mm supports only SM90 and SM100")
    torch.manual_seed(0)
    aq = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
    bq = torch.randn(2, 64, 48, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(64, 1, device="cuda", dtype=torch.float32)
    sb = torch.rand(2, 1, 48, device="cuda", dtype=torch.float32)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    expected = scaled_grouped_gemm(
        aq,
        bq,
        sa,
        sb,
        offs,
        torch.bfloat16,
        0,
        backend="reference",
    )
    actual = scaled_grouped_gemm(
        aq, bq, sa, sb, offs, torch.bfloat16, 0, backend="torch"
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@cuda_only
def test_forced_torch_scaled_grouped_gemm_rejects_unsupported_architecture():
    if torch.cuda.get_device_capability()[0] in (9, 10):
        pytest.skip("current architecture supports F.scaled_grouped_mm")
    aq = torch.ones(16, 16, device="cuda").to(torch.float8_e4m3fn)
    bq = torch.ones(1, 16, 16, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(16, 1, device="cuda")
    sb = torch.ones(1, 1, 16, device="cuda")
    offs = torch.tensor([16], device="cuda", dtype=torch.int32)

    with pytest.raises(KernelSelectionError, match="torch.*SM90 or SM100"):
        scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.bfloat16, 0, backend="torch")
