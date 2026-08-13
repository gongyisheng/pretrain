"""Backend-neutral public surface for the custom GEMM kernels.

Callers import from here and stay agnostic of which backend serves them. The
Triton implementations live in `src.kernel.triton.gemm`; the CuTe DSL ones in
`src.kernel.cute.gemm`.
"""

from src.kernel.triton.gemm import grouped_gemm, scaled_gemm, scaled_grouped_gemm

__all__ = ["grouped_gemm", "scaled_gemm", "scaled_grouped_gemm"]
