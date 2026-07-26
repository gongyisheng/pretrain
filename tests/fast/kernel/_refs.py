"""Eager reference implementations for kernel-level grouped-GEMM parity tests.

Ground truth only (plain torch ops) — this module must not import the kernels
under test, so a parity test always compares against an independent reference.
"""

import torch


def grouped_mm_ref(a, b, offs):
    """Forward reference for the grouped GEMM: torch._grouped_mm."""
    return torch._grouped_mm(a, b, offs=offs)


def wgrad_ref(a, grad_c, offs):
    """Weight-grad reference: per-group a[g].T @ grad_c[g] in fp32, shape (E, K, N)."""
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
