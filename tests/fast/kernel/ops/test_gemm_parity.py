import pytest
import torch

import src.kernel.ops.gemm  # noqa: F401  registers every backend
from src.kernel.registry import KERNEL_REGISTRY
from src.kernel.selector import _platform_for

E, M, K, N = 4, 128, 64, 96

SCALED_OPS = (
    "gemm.int8_scaled_mm",
    "gemm.fp8_scaled_mm",
    "gemm.mxfp8_scaled_mm",
)
GROUPED_SCALED_OPS = (
    "gemm.int8_scaled_grouped_mm",
    "gemm.fp8_scaled_grouped_mm",
    "gemm.mxfp8_scaled_grouped_mm",
)


def _operand_dtype(op):
    return torch.int8 if "int8" in op else torch.float8_e4m3fn


def _block_size(op):
    return 32 if "mxfp8" in op else 0


def _eligible(op, device):
    platform = _platform_for(device.type, device.index)
    return [
        spec
        for spec in KERNEL_REGISTRY.implementations(op)
        if spec.is_available(platform) and not spec.reference
    ]


def _pow2_scale(shape, device):
    # Power-of-two scales, unlike torch.ones, actually exercise the swizzle
    # permutation, transposes and block indexing: a wrong one still multiplies out
    # to the right answer when every scale is 1.0.
    return 2.0 ** torch.randint(-2, 3, shape, device=device, dtype=torch.float32)


def _scaled_inputs(op, device, grouped):
    dtype = _operand_dtype(op)
    block_size = _block_size(op)
    blocks = -(-K // block_size) if block_size else 1
    aq = torch.randint(-8, 8, (M, K), device=device).to(dtype)
    bq = torch.randint(-8, 8, (E, K, N) if grouped else (K, N), device=device).to(dtype)
    sa = _pow2_scale((M, blocks), device)
    sb = _pow2_scale((E, blocks, N) if grouped else (blocks, N), device)
    args = [aq, bq, sa, sb]
    if grouped:
        args.append(torch.arange(1, E + 1, device=device, dtype=torch.int32) * (M // E))
    return tuple(args) + (torch.bfloat16, block_size, None)


def _ragged_k_scaled_inputs(op, device):
    # 2D x 2D: the contraction itself is ragged (the wgrad layout), the third of
    # the three grouped scale layouts and, until now, untested by this harness.
    dtype = _operand_dtype(op)
    block_size = _block_size(op)
    group_k = K // E
    blocks_per_group = -(-group_k // block_size) if block_size else 1
    total_blocks = blocks_per_group * E
    aq = torch.randint(-8, 8, (M, K), device=device).to(dtype)
    bq = torch.randint(-8, 8, (K, N), device=device).to(dtype)
    sa = _pow2_scale((M, total_blocks), device)
    sb = _pow2_scale((total_blocks, N), device)
    offs = torch.arange(1, E + 1, device=device, dtype=torch.int32) * group_k
    return aq, bq, sa, sb, offs, torch.bfloat16, block_size, None


def _assert_all_eligible_match_eager(op, args):
    specs = _eligible(op, args[0].device)
    if not specs:
        pytest.skip(f"{op} has no eligible non-eager backend on this platform")
    eager = {s.backend: s for s in KERNEL_REGISTRY.implementations(op)}["eager"]
    reference = eager.fn(*args)

    for spec in specs:
        out = spec.fn(*args)
        torch.testing.assert_close(
            out.float(),
            reference.float(),
            rtol=2e-2,
            atol=2e-2,
            msg=lambda m, b=spec.backend: f"{op} backend {b} diverges from eager:\n{m}",
        )


@pytest.mark.parametrize("op", SCALED_OPS + GROUPED_SCALED_OPS)
def test_every_eligible_backend_matches_eager(op):
    # cuda when present, cpu otherwise: every optimized spec requires cuda
    # capabilities, so a CPU-only platform is simply never eligible and this
    # skips below rather than erroring on an unavailable device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _scaled_inputs(op, device, op in GROUPED_SCALED_OPS)
    _assert_all_eligible_match_eager(op, args)


@pytest.mark.parametrize("op", GROUPED_SCALED_OPS)
def test_every_eligible_backend_matches_eager_ragged_k(op):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _ragged_k_scaled_inputs(op, device)
    _assert_all_eligible_match_eager(op, args)


def test_cublaslt_is_absent_from_the_registry_when_the_extension_is_missing():
    import src.kernel.backends.cublaslt as backend

    registered = {
        spec.backend for spec in KERNEL_REGISTRY.implementations("gemm.mxfp8_scaled_mm")
    }

    assert ("cublaslt" in registered) is (backend._C is not None)
