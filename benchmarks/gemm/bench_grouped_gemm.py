"""Forward-only throughput: Triton grouped GEMM vs torch._grouped_mm.

Covers both configs' expert GEMMs across batch sizes, with a parity assert before
timing. Run: uv run python benchmarks/gemm/bench_grouped_gemm.py
"""

import sys
import time

import torch

sys.path.insert(0, ".")

from src.kernel.gemm import grouped_mm

SEQ = 1024

# (label, E, k, K, N) — the four expert GEMMs from the two configs.
CASES = [
    ("e512_k48 gate_up", 512, 48, 64, 384),
    ("e512_k48 down", 512, 48, 192, 64),
    ("e64_k6 gate_up", 64, 6, 512, 384),
    ("e64_k6 down", 64, 6, 192, 512),
]
BATCHES = [8, 16, 32, 64]


def _make(E, R, K, N):
    torch.manual_seed(0)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = w.mT  # (E, K, N)
    base = R // E
    counts = torch.full((E,), base, device="cuda", dtype=torch.int64)
    counts[-1] += R - int(counts.sum())
    offs = counts.cumsum(0).to(torch.int32)
    return a, b, offs


def _assert_parity(got, ref, chunk=500_000):
    # row-independent output → chunked compare bounds peak fp32/isclose memory
    for i in range(0, got.shape[0], chunk):
        torch.testing.assert_close(
            got[i : i + chunk].float(),
            ref[i : i + chunk].float(),
            rtol=2e-2,
            atol=2e-2,
        )


def _time(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(torch.cuda.get_device_name(0))
    print(
        f"{'case':20s} {'bs':>4s} {'rows/grp':>9s} {'torch_ms':>9s} {'triton_ms':>10s} {'speedup':>8s}"
    )
    for label, E, k, K, N in CASES:
        for bs in BATCHES:
            R = bs * SEQ * k
            a, b, offs = _make(E, R, K, N)
            ref = torch._grouped_mm(a, b, offs=offs)
            got = grouped_mm(a, b, offs)
            _assert_parity(got, ref)
            t_torch = _time(lambda: torch._grouped_mm(a, b, offs=offs))
            t_triton = _time(lambda: grouped_mm(a, b, offs))
            print(
                f"{label:20s} {bs:>4d} {R // E:>9d} {t_torch:>9.3f} "
                f"{t_triton:>10.3f} {t_torch / t_triton:>7.2f}x"
            )


if __name__ == "__main__":
    main()
