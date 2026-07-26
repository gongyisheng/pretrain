"""Parity of the Triton grouped GEMM prototype vs torch._grouped_mm (bf16)."""

import pytest
import torch

from tests.fast.kernel._refs import _ref_and_triton, _wgrad_ref

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton grouped GEMM is CUDA+bf16 only"
)


def test_parity_balanced_groups():
    ref, got = _ref_and_triton(counts=[128, 128, 128, 128], K=64, N=48)
    assert got.shape == ref.shape
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_parity_empty_and_uneven_groups():
    # includes an empty group (0 rows) and uneven sizes
    ref, got = _ref_and_triton(counts=[5, 0, 130, 41, 1], K=64, N=48)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_parity_single_group():
    ref, got = _ref_and_triton(counts=[200], K=192, N=64)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "K,N",
    [
        (64, 384),  # e512_k48 gate_up
        (192, 64),  # e512_k48 down
        (512, 384),  # e64_k6 gate_up
        (192, 512),  # e64_k6 down
    ],
)
def test_parity_config_shapes(K, N):
    # skewed load across 16 experts, summing to a realistic row count
    counts = [300, 5, 120, 0, 44, 210, 7, 90, 33, 150, 1, 260, 12, 80, 40, 400]
    ref, got = _ref_and_triton(counts=counts, K=K, N=N, seed=1)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_wgrad_parity_uneven_and_empty():
    from src.kernel.gemm import grouped_mm_wgrad

    torch.manual_seed(0)
    counts = [5, 0, 130, 41, 1]
    R, K, N = sum(counts), 64, 48
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    grad_c = torch.randn(R, N, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    ref = _wgrad_ref(a, grad_c, offs)
    got = grouped_mm_wgrad(a, grad_c, offs)
    assert got.shape == (len(counts), K, N)
    torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=2e-2)


def test_grouped_mm_fwd_bwd_matches_torch():
    from src.kernel.gemm import grouped_mm

    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a0 = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1

    def run(fn):
        a = a0.clone().requires_grad_(True)
        wv = w.clone().requires_grad_(True)
        c = fn(a, wv.mT, offs)
        # Use an explicit contiguous upstream grad instead of c.sum().backward().
        # d(sum(c))/dc is exactly ones_like(c), so this is mathematically identical,
        # but c.sum().backward() hands SumBackward's lazily-expanded (zero-stride)
        # grad straight to the op, and torch._grouped_mm's own CUDA backward on this
        # torch/GPU build (2.10.0+cu128, RTX 5090) mishandles that zero-stride grad
        # tensor and raises "Invalid strides/sizes" — a bug in the reference op
        # itself, reproducible even with plain torch._grouped_mm and no triton
        # involved. A materialized ones_like grad sidesteps it without changing what
        # is being checked.
        c.backward(torch.ones_like(c))
        return c, a.grad, wv.grad

    c_ref, ga_ref, gw_ref = run(lambda a, b, o: torch._grouped_mm(a, b, offs=o))
    c_got, ga_got, gw_got = run(lambda a, b, o: grouped_mm(a, b, o))
    torch.testing.assert_close(c_got.float(), c_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ga_got.float(), ga_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gw_got.float(), gw_ref.float(), rtol=2e-2, atol=2e-2)


def test_grouped_mm_compiles_fullgraph_and_matches_eager():
    from src.kernel.gemm import grouped_mm

    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a0 = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1

    def fn(a, b, o):
        return grouped_mm(a, b, o)

    def run(f):
        a = a0.clone().requires_grad_(True)
        wv = w.clone().requires_grad_(True)
        c = f(a, wv.mT, offs)
        c.sum().backward()
        return c, a.grad, wv.grad

    eager = run(fn)
    # fullgraph=True raises if the op graph-breaks
    compiled_fn = torch.compile(fn, fullgraph=True)
    comp = run(compiled_fn)
    for e, c in zip(eager, comp):
        torch.testing.assert_close(c.float(), e.float(), rtol=2e-2, atol=2e-2)


def test_dispatch_impl_selection_and_parity():
    from src.kernel import gemm

    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = w.mT
    ref = torch._grouped_mm(a, b, offs=offs)
    for impl in ("auto", "triton", "torch"):
        got = gemm.grouped_mm_dispatch(a, b, offs, impl)
        torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
    # torch path is exactly torch._grouped_mm
    assert torch.equal(gemm.grouped_mm_dispatch(a, b, offs, "torch"), ref)
    # invalid impl rejected
    import pytest as _pytest

    with _pytest.raises((ValueError, AssertionError)):
        gemm.grouped_mm_dispatch(a, b, offs, "nonsense")


def test_dispatch_triton_rejects_non_bf16():
    from src.kernel import gemm

    counts = [64, 130]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.float32)  # not bf16
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.float32) * 0.1
    b = w.mT
    with pytest.raises(ValueError):
        gemm.grouped_mm_dispatch(a, b, offs, "triton")
