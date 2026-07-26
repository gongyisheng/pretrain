"""
Eager reference implementations for kernel-level numerical parity tests.
"""

import torch


def _ref_and_triton(counts, K, N, seed=0):
    """Build (a, b=w.mT, offs) and return (torch._grouped_mm, triton_grouped_mm)."""
    from src.kernel.gemm import triton_grouped_mm

    torch.manual_seed(seed)
    E = len(counts)
    R = sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = (
        torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    )  # (E, out=N, in=K)
    b = w.mT  # (E, K, N), transposed view — non-contiguous
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    ref = torch._grouped_mm(a, b, offs=offs)
    got = triton_grouped_mm(a, b, offs)
    return ref, got


def _wgrad_ref(a, grad_c, offs):
    E = offs.shape[0]
    K, N = a.shape[1], grad_c.shape[1]
    out = torch.zeros(E, K, N, device=a.device, dtype=torch.float32)
    start = 0
    for g in range(E):
        end = int(offs[g])
        if end > start:
            out[g] = a[start:end].float().T @ grad_c[start:end].float()
        start = end
    return out
