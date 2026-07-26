"""Eager reference implementations for kernel-level grouped-GEMM parity tests.

Ground truth only (plain torch ops) — this module must not import the kernels
under test, so a parity test always compares against an independent reference.
"""

import torch


def grouped_gemm_ref(a, b, offs):
    """Forward reference for the grouped GEMM: torch._grouped_mm."""
    return torch._grouped_mm(a, b, offs=offs)


def grouped_gemm_wgrad_ref(a, grad_c, offs):
    """Weight-grad reference: grad_b[g] = a[g].T @ grad_c[g], shape (E, K, N)."""
    return torch._grouped_mm(a.mT, grad_c, offs=offs)
