"""Native cuBLASLt GEMM operator tests."""

import torch

import src.kernel.backends.cublaslt._C  # noqa: F401


def test_native_operator_has_meta_implementation():
    a = torch.empty((129, 144), device="meta", dtype=torch.float8_e4m3fn)
    b = torch.empty_strided(
        (144, 160), (1, 144), device="meta", dtype=torch.float8_e4m3fn
    )
    scale_a = torch.empty(2048, device="meta", dtype=torch.float8_e8m0fnu)
    scale_b = torch.empty(2048, device="meta", dtype=torch.float8_e8m0fnu)
    out = torch.ops.aot_kernel.scaled_gemm_mxfp8(a, b, scale_a, scale_b, None)
    assert out.shape == (129, 160)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "meta"
