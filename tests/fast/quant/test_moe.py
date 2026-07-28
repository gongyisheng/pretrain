import pytest
import torch

from src.quant import moe
from src.quant.quantize import (  # noqa: F401  (effective_block_size used in Task 3 tests)
    effective_block_size,
    fake_quantize_operand,
)
from src.utils.config import QuantConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="scaled grouped GEMM is CUDA only"
)


def _cfg(recipe="fp8", scaling="rowwise"):
    return QuantConfig(
        enabled=True, dtype={"recipe": recipe}, scaling={"recipe": scaling}
    )


def _make(counts, K, N, seed=0):
    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return a, b, offs


def _fwd_oracle(a, b, offs, cfg):
    """Dequant-both fp32 grouped matmul (fake-quant reference)."""
    gran = cfg.scaling["granularity"]
    bs = cfg.scaling.get("block_size", 0)
    a_fq = fake_quantize_operand(a, -1, gran, bs, cfg.dtype["act"]).float()
    out = torch.zeros(a.shape[0], b.shape[2], device=a.device, dtype=torch.float32)
    start = 0
    for g, end in enumerate(offs.tolist()):
        if end > start:
            bg = fake_quantize_operand(b[g], 0, gran, bs, cfg.dtype["weight"]).float()
            out[start:end] = a_fq[start:end] @ bg
        start = end
    return out


@pytest.mark.parametrize("scaling", ["rowwise", "tensorwise"])
def test_scaled_grouped_gemm_forward_matches_oracle(scaling):
    a, b, offs = _make([128, 0, 130, 41], K=64, N=48)
    cfg = _cfg("fp8", scaling)
    got = moe.scaled_grouped_gemm(a, b, offs, cfg)
    ref = _fwd_oracle(a, b, offs, cfg)
    assert got.shape == (a.shape[0], b.shape[2])
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < 5e-2, rel
