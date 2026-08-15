"""GEMM backend parity against the eager correctness backend."""

import pytest


import torch


from src.kernel.selector import KernelSelectionError


from src.kernel.ops.gemm import (
    grouped_gemm,
    scaled_gemm,
    scaled_grouped_gemm,
)


from src.quant.quantize import quantize_operand


def grouped_gemm_eager(a, b, offs, bias=None):
    return grouped_gemm(a, b, offs, bias=bias, backend="eager")


def scaled_gemm_eager(aq, bq, sa, sb, block_size, bias=None):
    return scaled_gemm(
        aq,
        bq,
        sa,
        sb,
        torch.float32,
        block_size,
        bias=bias,
        backend="eager",
    )


def scaled_grouped_gemm_eager(
    aq,
    bq,
    sa,
    sb,
    offs,
    block_size,
    bias=None,
):
    return scaled_grouped_gemm(
        aq,
        bq,
        sa,
        sb,
        offs,
        torch.float32,
        block_size,
        bias=bias,
        backend="eager",
    )


def test_public_gemm_import_surface():
    from src.kernel.ops import grouped_gemm, scaled_gemm, scaled_grouped_gemm

    assert callable(grouped_gemm)
    assert callable(scaled_gemm)
    assert callable(scaled_grouped_gemm)


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton GEMM kernels are CUDA only"
)


def _scaling(gran, bs=0, scale_dtype=None):
    return {
        "granularity": gran,
        "block_shape": (1, bs) if bs else (0, 0),
        "scale_dtype": scale_dtype,
    }


@cuda_only
def test_backend_selection_and_parity():
    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = w.mT
    ref = grouped_gemm_eager(a, b, offs)
    for backend in ("auto", "triton", "torch"):
        got = grouped_gemm(a, b, offs, None, backend)
        torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        grouped_gemm(a, b, offs, backend="torch").float(),
        ref.float(),
        rtol=2e-2,
        atol=2e-2,
    )
    # invalid backend rejected
    with pytest.raises((KernelSelectionError)):
        grouped_gemm(a, b, offs, backend="nonsense")


@cuda_only
def test_grouped_gemm_auto_compiles_fullgraph():
    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    fn = torch.compile(
        lambda a, b, o: grouped_gemm(a, b, o, backend="auto"), fullgraph=True
    )
    out = fn(a, w.mT, offs)
    torch.testing.assert_close(
        out.float(),
        grouped_gemm_eager(a, w.mT, offs).float(),
        rtol=2e-2,
        atol=2e-2,
    )


@cuda_only
def test_grouped_gemm_auto_selects_triton_on_this_arch():
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
    auto = grouped_gemm(a, b, offs, backend="auto")
    assert torch.equal(auto, grouped_gemm(a, b, offs, backend="triton"))


@cuda_only
def test_grouped_gemm_auto_uses_eager_for_non_bf16():
    counts = [64, 130]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.float32)
    b = torch.randn(len(counts), N, K, device="cuda", dtype=torch.float32).mT
    assert torch.equal(grouped_gemm(a, b, offs), grouped_gemm_eager(a, b, offs))


@cuda_only
def test_dispatch_triton_rejects_non_bf16():
    counts = [64, 130]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(R, K, device="cuda", dtype=torch.float32)  # not bf16
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.float32) * 0.1
    b = w.mT
    with pytest.raises(KernelSelectionError):
        grouped_gemm(a, b, offs, backend="triton")


def rand(rows, columns):
    return torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)


SCALED_LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")


_SCALED_COUNTS = [64, 0, 130, 46]  # sums to 240; the empty group is deliberate


def _make_scaled_layout(
    layout, counts, gran, bs, fmt, M=32, K=64, N=48, seed=0, scale_dtype=None
):
    """Quantized operands + scales for one ragged layout of the scaled grouped GEMM.

    Scales follow the kernel's invariant: each mirrors its operand's axis order with
    the contraction axis replaced by the scale-block axis. `bs` 0 (row/tensorwise)
    means one block per contraction segment.
    """
    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    scaling = _scaling(gran, bs, scale_dtype)

    def rand(dim0, dim1, dim2=None):
        shape = (dim0, dim1) if dim2 is None else (dim0, dim1, dim2)
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1

    if layout == "ragged_m":  # (R,K) x (E,K,N) -> (R,N)
        aq, sa = quantize_operand(rand(R, K), -1, fmt, scaling)
        b = torch.stack([rand(K, N) for _ in range(E)])
        bq, sb = quantize_operand(b, -2, fmt, scaling)
        return aq, bq, sa, sb, offs, bs

    if layout == "ragged_k":  # (M,R) x (R,N) -> (E,M,N)
        aq, sa = quantize_operand(
            rand(R, M), -2, fmt, scaling, offs=offs, ragged_dim=-2
        )
        bq, sb = quantize_operand(
            rand(R, N), -2, fmt, scaling, offs=offs, ragged_dim=-2
        )
        return aq.mT, bq, sa.mT, sb, offs, bs

    if layout == "ragged_n":  # (E,M,K) x (K,R) -> (M,R)
        a = torch.stack([rand(M, K) for _ in range(E)])
        aq, sa = quantize_operand(a, -1, fmt, scaling)
        bqT, sbT = quantize_operand(
            rand(R, K), -1, fmt, scaling, offs=offs, ragged_dim=-2
        )
        return aq, bqT.mT, sa, sbT.mT, offs, bs

    raise ValueError(layout)


@cuda_only
@pytest.mark.parametrize("layout", SCALED_LAYOUTS)
def test_scaled_grouped_layouts_compile_fullgraph(layout):
    aq, bq, sa, sb, offs, kbs = _make_scaled_layout(
        layout, _SCALED_COUNTS, "blockwise", 32, "fp8_e4m3", seed=4
    )
    fn = torch.compile(
        lambda: scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.bfloat16, kbs),
        fullgraph=True,
    )
    assert torch.isfinite(fn()).all()


@cuda_only
def test_compiles_fullgraph():
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, "fp8_e4m3", _scaling("blockwise", 32))
    bq, sb = quantize_operand(b, -2, "fp8_e4m3", _scaling("blockwise", 32))
    fn = torch.compile(
        lambda: scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 32), fullgraph=True
    )
    assert torch.isfinite(fn()).all()


@cuda_only
def test_scaled_gemm_2d_scales_compile_fullgraph():
    """repeat_interleave with a constant tile must not graph-break."""
    a, b = rand(256, 512), rand(512, 128)
    scaling = {
        "granularity": "blockwise",
        "block_shape": (32, 32),
        "scale_dtype": "fp32",
    }
    aq, sa = quantize_operand(a, -1, "fp8_e4m3", scaling)
    bq, sb = quantize_operand(b, -2, "fp8_e4m3", scaling)
    fn = torch.compile(
        lambda: scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 32), fullgraph=True
    )
    assert torch.isfinite(fn()).all()


@cuda_only
def test_cublaslt_mxfp8_scaled_gemm_compiles_fullgraph_bitwise():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuBLASLt MXFP8 GEMM requires SM100 or newer")
    torch.manual_seed(0)
    a = torch.randn(129, 144, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(144, 160, device="cuda", dtype=torch.bfloat16)
    scaling = _scaling("blockwise", 32, "fp8_e8m0")
    aq, sa = quantize_operand(a, -1, "fp8_e4m3", scaling)
    bq, sb = quantize_operand(b, -2, "fp8_e4m3", scaling)

    eager = scaled_gemm(
        aq,
        bq,
        sa,
        sb,
        torch.bfloat16,
        32,
        scale_dtype="fp8_e8m0",
        backend="eager",
    )
    compiled = torch.compile(
        lambda aq, bq, sa, sb: scaled_gemm(
            aq,
            bq,
            sa,
            sb,
            torch.bfloat16,
            32,
            scale_dtype="fp8_e8m0",
            backend="cublaslt",
        ),
        fullgraph=True,
    )
    actual = compiled(aq, bq, sa, sb)
    difference = actual.float() - eager.float()
    relative_error = (difference.norm() / eager.float().norm()).item()
    maximum_error = difference.abs().max().item()

    assert torch.equal(actual, eager), (relative_error, maximum_error)


def test_grouped_gemm_auto_falls_back_to_eager_for_float32():
    a = torch.randn(4, 8, device="cpu")
    b = torch.randn(1, 8, 6, device="cpu")
    offs = torch.tensor([4], device="cpu")
    got = grouped_gemm(a, b, offs)
    expected = grouped_gemm_eager(a, b, offs)
    torch.testing.assert_close(got, expected)


def test_grouped_gemm_eagererence_keeps_misaligned_groups_separate():
    a = torch.randn(5, 8, device="cpu")
    b = torch.randn(2, 8, 6, device="cpu")
    offs = torch.tensor([2, 5], dtype=torch.int32, device="cpu")
    got = grouped_gemm(a, b, offs)
    expected = grouped_gemm_eager(a, b, offs)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("backend", ["auto", "torch", "triton"])
def test_grouped_gemm_rejects_mismatched_bias_dtype(backend):
    a = torch.randn(4, 8, device="cpu", dtype=torch.float32)
    b = torch.randn(1, 8, 6, device="cpu", dtype=torch.float32)
    offs = torch.tensor([4], device="cpu")
    bias = torch.randn(1, 6, device="cpu", dtype=torch.float64)
    with pytest.raises(KernelSelectionError, match="bias dtype.*match"):
        grouped_gemm(a, b, offs, bias=bias, backend=backend)


def test_grouped_gemm_eagererence_bias_preserves_output_dtype():
    a = torch.randn(4, 8, device="cpu", dtype=torch.float32)
    b = torch.randn(1, 8, 6, device="cpu", dtype=torch.float32)
    offs = torch.tensor([4], device="cpu")
    bias = torch.randn(1, 6, device="cpu", dtype=torch.float32)
    out = grouped_gemm(a, b, offs, bias=bias, backend="eager")
    assert out.dtype == a.dtype


def test_forced_triton_does_not_fall_back_for_float32():
    a = torch.randn(4, 8, device="cpu")
    b = torch.randn(1, 8, 6, device="cpu")
    offs = torch.tensor([4], device="cpu")
    with pytest.raises(KernelSelectionError, match="triton.*CUDA"):
        grouped_gemm(a, b, offs, backend="triton")


def test_scaled_gemm_auto_falls_back_to_eager_on_cpu():
    aq = torch.tensor([[1, -2, 3, 4], [-1, 2, 0, 3]], dtype=torch.int8, device="cpu")
    bq = torch.tensor(
        [[1, 2, -1], [0, -2, 3], [2, 1, 1], [-1, 0, 2]],
        dtype=torch.int8,
        device="cpu",
    )
    sa = torch.tensor([[0.5, 0.25], [0.75, 0.5]], device="cpu")
    sb = torch.tensor([[1.0, 0.5, 0.25], [0.5, 1.0, 2.0]], device="cpu")
    got = scaled_gemm(aq, bq, sa, sb, torch.float32, 2)
    expected = scaled_gemm_eager(aq, bq, sa, sb, 2)
    torch.testing.assert_close(got, expected)


def test_scaled_grouped_gemm_auto_falls_back_to_eager_on_cpu():
    aq = torch.tensor(
        [[1, -2, 3, 4], [-1, 2, 0, 3], [2, 1, -1, 0], [0, 3, 2, -2]],
        dtype=torch.int8,
        device="cpu",
    )
    bq = torch.tensor(
        [
            [[1, 2, -1], [0, -2, 3], [2, 1, 1], [-1, 0, 2]],
            [[-1, 1, 2], [2, 0, -2], [1, 3, 0], [0, -1, 1]],
        ],
        dtype=torch.int8,
        device="cpu",
    )
    sa = torch.tensor([[0.5, 0.25], [0.75, 0.5], [1.0, 0.5], [0.25, 1.0]], device="cpu")
    sb = torch.tensor(
        [
            [[1.0, 0.5, 0.25], [0.5, 1.0, 2.0]],
            [[0.25, 1.0, 0.5], [2.0, 0.5, 1.0]],
        ],
        device="cpu",
    )
    offs = torch.tensor([1, 4], dtype=torch.int32, device="cpu")
    got = scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.float32, 2)
    expected = scaled_grouped_gemm_eager(aq, bq, sa, sb, offs, 2)
    torch.testing.assert_close(got, expected)
