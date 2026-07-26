"""Parity of the Triton grouped GEMM prototype vs torch._grouped_mm (bf16)."""

import pytest
import torch

from tests.fast.kernel._refs import grouped_gemm_ref, grouped_gemm_wgrad_ref

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton grouped GEMM is CUDA+bf16 only"
)


def _make(counts, K, N, seed=0):
    """Build (a (R,K), b=w.mT (E,K,N) transposed view, offs (E,) int32) on CUDA/bf16."""
    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    b = (torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.1).mT
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return a, b, offs


def test_parity_balanced_groups():
    from src.kernel.gemm import grouped_gemm

    a, b, offs = _make([128, 128, 128, 128], K=64, N=48)
    got = grouped_gemm(a, b, offs)
    ref = grouped_gemm_ref(a, b, offs)
    assert got.shape == ref.shape
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_parity_empty_and_uneven_groups():
    from src.kernel.gemm import grouped_gemm

    # includes an empty group (0 rows) and uneven sizes
    a, b, offs = _make([5, 0, 130, 41, 1], K=64, N=48)
    got = grouped_gemm(a, b, offs)
    torch.testing.assert_close(
        got.float(), grouped_gemm_ref(a, b, offs).float(), rtol=2e-2, atol=2e-2
    )


def test_parity_single_group():
    from src.kernel.gemm import grouped_gemm

    a, b, offs = _make([200], K=192, N=64)
    got = grouped_gemm(a, b, offs)
    torch.testing.assert_close(
        got.float(), grouped_gemm_ref(a, b, offs).float(), rtol=2e-2, atol=2e-2
    )


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
    from src.kernel.gemm import grouped_gemm

    # skewed load across 16 experts, summing to a realistic row count
    counts = [300, 5, 120, 0, 44, 210, 7, 90, 33, 150, 1, 260, 12, 80, 40, 400]
    a, b, offs = _make(counts, K=K, N=N, seed=1)
    got = grouped_gemm(a, b, offs)
    torch.testing.assert_close(
        got.float(), grouped_gemm_ref(a, b, offs).float(), rtol=2e-2, atol=2e-2
    )


def test_wgrad_parity_uneven_and_empty():
    from src.kernel.gemm import _grouped_gemm_wgrad

    torch.manual_seed(0)
    counts = [5, 0, 130, 41, 1]
    R, K, N = sum(counts), 64, 48
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    grad_c = torch.randn(R, N, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    ref = grouped_gemm_wgrad_ref(a, grad_c, offs)
    got = _grouped_gemm_wgrad(a, grad_c, offs)
    assert got.shape == (len(counts), K, N)
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
def test_wgrad_parity_config_shapes(K, N):
    from src.kernel.gemm import _grouped_gemm_wgrad

    torch.manual_seed(1)
    counts = [300, 5, 120, 0, 44, 210, 7, 90, 33, 150, 1, 260, 12, 80, 40, 400]
    R = sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    grad_c = torch.randn(R, N, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    ref = grouped_gemm_wgrad_ref(a, grad_c, offs)
    got = _grouped_gemm_wgrad(a, grad_c, offs)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_grouped_gemm_fwd_bwd_matches_torch():
    from src.kernel.gemm import _grouped_gemm

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
        c.backward(torch.ones_like(c))
        return c, a.grad, wv.grad

    c_ref, ga_ref, gw_ref = run(lambda a, b, o: torch._grouped_mm(a, b, offs=o))
    c_got, ga_got, gw_got = run(lambda a, b, o: _grouped_gemm(a, b, o))
    torch.testing.assert_close(c_got.float(), c_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ga_got.float(), ga_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gw_got.float(), gw_ref.float(), rtol=2e-2, atol=2e-2)


def test_grouped_gemm_compiles_fullgraph_and_matches_eager():
    from src.kernel.gemm import _grouped_gemm

    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a0 = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1

    def fn(a, b, o):
        return _grouped_gemm(a, b, o)

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
        got = gemm.grouped_gemm(a, b, offs, impl)
        torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
    # torch path is exactly torch._grouped_mm
    assert torch.equal(gemm.grouped_gemm(a, b, offs, "torch"), ref)
    # invalid impl rejected
    import pytest as _pytest

    with _pytest.raises((ValueError, AssertionError)):
        gemm.grouped_gemm(a, b, offs, "nonsense")


def test_grouped_gemm_auto_compiles_fullgraph():
    from src.kernel.gemm import grouped_gemm

    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    fn = torch.compile(lambda a, b, o: grouped_gemm(a, b, o, "auto"), fullgraph=True)
    out = fn(a, w.mT, offs)
    torch.testing.assert_close(
        out.float(),
        torch._grouped_mm(a, w.mT, offs=offs).float(),
        rtol=2e-2,
        atol=2e-2,
    )


def test_grouped_gemm_auto_selects_triton_on_this_arch():
    from src.kernel import gemm

    major = torch.cuda.get_device_capability()[0]
    if major in (9, 10):
        pytest.skip("fused torch path available; auto uses torch here")
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    b = (torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1).mT
    # auto must take the Triton path here → bitwise-identical to forcing triton,
    # and (almost certainly) NOT bitwise-equal to the torch fallback.
    auto = gemm.grouped_gemm(a, b, offs, "auto")
    assert torch.equal(auto, gemm.grouped_gemm(a, b, offs, "triton"))


def test_grouped_gemm_auto_uses_torch_for_non_bf16():
    from src.kernel import gemm

    counts = [64, 130]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.float32)
    b = torch.randn(len(counts), N, K, device="cuda", dtype=torch.float32).mT
    assert torch.equal(
        gemm.grouped_gemm(a, b, offs, "auto"), torch._grouped_mm(a, b, offs=offs)
    )


def test_dispatch_triton_rejects_non_bf16():
    from src.kernel import gemm

    counts = [64, 130]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.float32)  # not bf16
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.float32) * 0.1
    b = w.mT
    with pytest.raises(ValueError):
        gemm.grouped_gemm(a, b, offs, "triton")
