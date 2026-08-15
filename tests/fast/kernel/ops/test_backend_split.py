import pytest
import torch
import torch.nn.functional as F

from src.kernel.backends.torch import gemm as torch_gemm
from src.kernel.backends.triton import gemm as triton_gemm
from src.kernel.ops.gemm import grouped_gemm, scaled_gemm, scaled_grouped_gemm
from src.kernel.registry import KERNEL_REGISTRY
from src.kernel.selector import KernelSelectionError


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimized Torch GEMM requires CUDA"
)


class _TensorMeta:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: str = "cuda:7",
        contiguous: bool = True,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.ndim = len(shape)
        self.is_cuda = self.device.type == "cuda"
        self._contiguous = contiguous
        if strides is None:
            running = 1
            calculated = []
            for size in reversed(shape):
                calculated.append(running)
                running *= size
            self._strides = tuple(reversed(calculated))
        else:
            self._strides = strides

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self, dim: int | None = None):
        if dim is None:
            return self._strides
        return self._strides[dim]

    def element_size(self) -> int:
        return self.dtype.itemsize

    def numel(self) -> int:
        total = 1
        for size in self.shape:
            total *= size
        return total


def _grouped_metadata(device: str = "cuda:7"):
    return (
        _TensorMeta((64, 64), torch.bfloat16, device),
        _TensorMeta((2, 64, 48), torch.bfloat16, device),
        _TensorMeta((2,), torch.int32, device),
        None,
    )


def _scaled_metadata(device: str = "cuda:7"):
    return (
        _TensorMeta((64, 64), torch.float8_e4m3fn, device),
        _TensorMeta((64, 48), torch.float8_e4m3fn, device),
        _TensorMeta((64, 1), torch.float32, device),
        _TensorMeta((1, 48), torch.float32, device),
        torch.bfloat16,
        0,
        None,
        None,
    )


def _scaled_grouped_metadata(device: str = "cuda:7"):
    return (
        _TensorMeta((64, 64), torch.float8_e4m3fn, device),
        _TensorMeta((2, 64, 48), torch.float8_e4m3fn, device),
        _TensorMeta((64, 1), torch.float32, device),
        _TensorMeta((2, 1, 48), torch.float32, device),
        _TensorMeta((2,), torch.int32, device),
        torch.bfloat16,
        0,
        None,
        None,
    )


@pytest.mark.parametrize("op", ["gemm.grouped", "gemm.scaled", "gemm.scaled_grouped"])
def test_gemm_registry_contains_eager_torch_and_triton(op):
    implementations = KERNEL_REGISTRY.implementations(op)
    backends = {spec.backend for spec in implementations}
    assert backends == {"eager", "torch", "triton"}
    priorities = {spec.backend: spec.priority for spec in implementations}
    assert priorities["triton"] > priorities["torch"] > priorities["eager"]
    for spec in implementations:
        assert callable(spec.can_implement)
        assert not hasattr(spec, "availability")
        assert not hasattr(spec, "eligibility")


@pytest.mark.parametrize(
    ("predicate", "args_factory", "actual", "expected"),
    [
        (torch_gemm.can_implement_grouped_gemm, _grouped_metadata, (8, 9), True),
        (torch_gemm.can_implement_grouped_gemm, _grouped_metadata, (9, 0), True),
        (torch_gemm.can_implement_grouped_gemm, _grouped_metadata, (10, 0), True),
        (torch_gemm.can_implement_grouped_gemm, _grouped_metadata, (12, 0), True),
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata, (8, 9), True),
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata, (9, 0), True),
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata, (10, 0), True),
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata, (12, 0), True),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            (8, 9),
            False,
        ),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            (9, 0),
            True,
        ),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            (10, 0),
            True,
        ),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            (12, 0),
            False,
        ),
    ],
)
def test_torch_can_implement_architecture_matrix_uses_operand_device(
    monkeypatch: pytest.MonkeyPatch,
    predicate,
    args_factory,
    actual: tuple[int, int],
    expected: bool,
):
    seen = []

    def get_device_capability(device: torch.device) -> tuple[int, int]:
        seen.append(device)
        return actual

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)

    result = predicate(args_factory(), {})

    assert result.ok is expected
    assert seen == [torch.device("cuda:7")]
    if actual == (12, 0) and predicate is torch_gemm.can_implement_scaled_grouped_gemm:
        assert result.reason is not None
        assert "SM120" in result.reason
        assert "SM90 or SM100" in result.reason


@pytest.mark.parametrize(
    ("attribute", "predicate", "args_factory"),
    [
        ("grouped_mm", torch_gemm.can_implement_grouped_gemm, _grouped_metadata),
        ("scaled_mm", torch_gemm.can_implement_scaled_gemm, _scaled_metadata),
        (
            "scaled_grouped_mm",
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
        ),
    ],
)
def test_torch_can_implement_requires_callable_functional(
    monkeypatch: pytest.MonkeyPatch, attribute: str, predicate, args_factory
):
    monkeypatch.setattr(F, attribute, None)

    result = predicate(args_factory(), {})

    assert not result.ok
    assert result.reason is not None
    assert attribute in result.reason
    assert "callable" in result.reason


@pytest.mark.parametrize(
    ("predicate", "args_factory"),
    [
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata),
        (torch_gemm.can_implement_scaled_grouped_gemm, _scaled_grouped_metadata),
    ],
)
def test_torch_scaled_can_implement_requires_rowwise_scaling_enum(
    monkeypatch: pytest.MonkeyPatch, predicate, args_factory
):
    monkeypatch.setattr(F, "ScalingType", type("ScalingType", (), {}))

    result = predicate(args_factory(), {})

    assert not result.ok
    assert result.reason is not None
    assert "ScalingType.RowWise" in result.reason


def test_torch_scaled_grouped_rejects_noncontiguous_offsets():
    args = list(_scaled_grouped_metadata())
    args[4] = _TensorMeta((2,), torch.int32, contiguous=False, strides=(2,))

    result = torch_gemm.can_implement_scaled_grouped_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "contiguous" in result.reason


def test_triton_accepts_sm120_scaled_grouped_call_rejected_by_torch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = _scaled_grouped_metadata()

    torch_result = torch_gemm.can_implement_scaled_grouped_gemm(args, {})
    triton_result = triton_gemm.can_implement_scaled_grouped_gemm(args, {})

    assert not torch_result.ok
    assert triton_result.ok


def test_triton_grouped_accepts_sm90_with_supported_bias(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    args = list(_grouped_metadata())
    args[3] = _TensorMeta((2, 48), torch.bfloat16)

    torch_result = torch_gemm.can_implement_grouped_gemm(tuple(args), {})
    triton_result = triton_gemm.can_implement_grouped_gemm(tuple(args), {})

    assert not torch_result.ok
    assert triton_result.ok


def test_triton_grouped_rejects_noncontiguous_offsets():
    args = list(_grouped_metadata())
    args[2] = _TensorMeta((2,), torch.int32, contiguous=False, strides=(2,))

    result = triton_gemm.can_implement_grouped_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "contiguous" in result.reason


def test_triton_scaled_rejects_cross_device_scale():
    args = list(_scaled_metadata())
    args[2] = _TensorMeta((64, 1), torch.float32, "cuda:6")

    result = triton_gemm.can_implement_scaled_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "same device" in result.reason


def test_triton_scaled_grouped_rejects_unknown_scale_dtype():
    args = list(_scaled_grouped_metadata())
    args[8] = "unknown"

    result = triton_gemm.can_implement_scaled_grouped_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "scale_dtype" in result.reason


def test_eager_grouped_gemm_is_a_cpu_oracle_with_gradients():
    a = torch.randn(5, 8, device="cpu", dtype=torch.float32, requires_grad=True)
    b = torch.randn(2, 8, 6, device="cpu", dtype=torch.float32, requires_grad=True)
    offs = torch.tensor([2, 5], device="cpu", dtype=torch.int32)

    out = grouped_gemm(a, b, offs, backend="eager")
    expected = torch.cat((a[:2] @ b[0], a[2:] @ b[1]))

    torch.testing.assert_close(out, expected)
    out.sum().backward()
    assert a.grad is not None
    assert b.grad is not None


def test_forced_torch_grouped_gemm_rejects_cpu_without_fallback():
    a = torch.randn(4, 8, device="cpu")
    b = torch.randn(1, 8, 6, device="cpu")
    offs = torch.tensor([4], device="cpu", dtype=torch.int32)

    with pytest.raises(KernelSelectionError, match="torch.*CUDA"):
        grouped_gemm(a, b, offs, backend="torch")


def test_auto_scaled_gemm_uses_eager_for_cpu_blockwise_inputs():
    aq = torch.tensor([[1, -2, 3, 4], [-1, 2, 0, 3]], device="cpu", dtype=torch.int8)
    bq = torch.tensor(
        [[1, 2, -1], [0, -2, 3], [2, 1, 1], [-1, 0, 2]],
        device="cpu",
        dtype=torch.int8,
    )
    sa = torch.tensor([[0.5, 0.25], [0.75, 0.5]], device="cpu")
    sb = torch.tensor([[1.0, 0.5, 0.25], [0.5, 1.0, 2.0]], device="cpu")

    expected = scaled_gemm(aq, bq, sa, sb, torch.float32, 2, backend="eager")
    actual = scaled_gemm(aq, bq, sa, sb, torch.float32, 2)

    torch.testing.assert_close(actual, expected)


@cuda_only
def test_torch_grouped_gemm_matches_eager_for_supported_inputs():
    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    expected = grouped_gemm(a, b, offs, backend="eager")
    actual = grouped_gemm(a, b, offs, backend="torch")

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@cuda_only
def test_forced_torch_grouped_gemm_rejects_unsupported_bias():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
    bias = torch.randn(2, 48, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(KernelSelectionError, match="torch.*does not support bias"):
        grouped_gemm(a, b, offs, bias=bias, backend="torch")


@cuda_only
def test_torch_scaled_gemm_matches_eager_for_supported_inputs():
    torch.manual_seed(0)
    aq = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
    bq = torch.randn(64, 48, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(64, 1, device="cuda", dtype=torch.float32)
    sb = torch.rand(1, 48, device="cuda", dtype=torch.float32)
    bias = torch.randn(48, device="cuda", dtype=torch.bfloat16)

    expected = scaled_gemm(
        aq, bq, sa, sb, torch.bfloat16, 0, bias=bias, backend="eager"
    )
    actual = scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 0, bias=bias, backend="torch")

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_forced_torch_scaled_gemm_rejects_unsupported_cpu_inputs():
    aq = torch.ones(16, 16, device="cpu", dtype=torch.int8)
    bq = torch.ones(16, 16, device="cpu", dtype=torch.int8)
    sa = torch.ones(16, 1, device="cpu")
    sb = torch.ones(1, 16, device="cpu")

    with pytest.raises(KernelSelectionError, match="torch.*CUDA"):
        scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 0, backend="torch")


@cuda_only
def test_non_autograd_torch_grouped_gemm_is_not_selected_for_direct_autograd():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    with pytest.raises(
        KernelSelectionError, match="does not support ordinary autograd"
    ):
        grouped_gemm(a, b, offs, backend="torch")

    out = grouped_gemm(a, b, offs)
    out.float().sum().backward()
    assert a.grad is not None
    assert b.grad is not None


@cuda_only
def test_torch_scaled_grouped_gemm_matches_eager_when_architecture_supports_it():
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("F.scaled_grouped_mm supports only SM90 and SM100")
    torch.manual_seed(0)
    aq = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
    bq = torch.randn(2, 64, 48, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(64, 1, device="cuda", dtype=torch.float32)
    sb = torch.rand(2, 1, 48, device="cuda", dtype=torch.float32)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    expected = scaled_grouped_gemm(
        aq,
        bq,
        sa,
        sb,
        offs,
        torch.bfloat16,
        0,
        backend="eager",
    )
    actual = scaled_grouped_gemm(
        aq, bq, sa, sb, offs, torch.bfloat16, 0, backend="torch"
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@cuda_only
def test_forced_torch_scaled_grouped_gemm_rejects_unsupported_architecture():
    if torch.cuda.get_device_capability()[0] in (9, 10):
        pytest.skip("current architecture supports F.scaled_grouped_mm")
    aq = torch.ones(16, 16, device="cuda").to(torch.float8_e4m3fn)
    bq = torch.ones(1, 16, 16, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(16, 1, device="cuda")
    sb = torch.ones(1, 1, 16, device="cuda")
    offs = torch.tensor([16], device="cuda", dtype=torch.int32)

    with pytest.raises(KernelSelectionError, match="torch.*SM90 or SM100"):
        scaled_grouped_gemm(aq, bq, sa, sb, offs, torch.bfloat16, 0, backend="torch")


@cuda_only
def test_forced_torch_scaled_gemm_rejects_non_row_major_a():
    aq = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn).mT
    bq = torch.randn(64, 48, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(64, 1, device="cuda")
    sb = torch.ones(1, 48, device="cuda")

    with pytest.raises(KernelSelectionError, match="torch.*row-major"):
        scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 0, backend="torch")


@cuda_only
def test_torch_scaled_grouped_can_implement_rejects_non_row_major_a_before_arch():

    aq = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn).mT
    bq = torch.randn(2, 64, 48, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(64, 1, device="cuda")
    sb = torch.ones(2, 1, 48, device="cuda")
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    support = torch_gemm.can_implement_scaled_grouped_gemm(
        (aq, bq, sa, sb, offs, torch.bfloat16, 0, None, None), {}
    )

    assert not support.ok
    assert support.reason is not None and "row-major" in support.reason


@cuda_only
def test_torch_scaled_grouped_can_implement_rejects_unaligned_dimensions():

    aq = torch.randn(64, 48, device="cuda").to(torch.float8_e4m3fn)
    bq = torch.randn(2, 48, 50, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(64, 1, device="cuda")
    sb = torch.ones(2, 1, 50, device="cuda")
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    support = torch_gemm.can_implement_scaled_grouped_gemm(
        (aq, bq, sa, sb, offs, torch.bfloat16, 0, None, None), {}
    )

    assert not support.ok
    assert support.reason is not None and "multiples of 16" in support.reason


@cuda_only
@pytest.mark.parametrize("layout", ["ragged_m", "ragged_k", "ragged_n"])
def test_forced_torch_grouped_gemm_rejects_rank_specific_contraction_mismatch(
    layout,
):
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
    if layout == "ragged_m":
        a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 32, 48, device="cuda", dtype=torch.bfloat16)
    elif layout == "ragged_k":
        a = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(32, 48, device="cuda", dtype=torch.bfloat16)
    else:
        a = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(KernelSelectionError, match="torch.*contraction"):
        grouped_gemm(a, b, offs, backend="torch")


@cuda_only
def test_forced_torch_grouped_gemm_rejects_group_offset_count_mismatch():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([64], device="cuda", dtype=torch.int32)

    with pytest.raises(KernelSelectionError, match="torch.*group count.*offset"):
        grouped_gemm(a, b, offs, backend="torch")


@cuda_only
@pytest.mark.parametrize("offset_layout", ["rank", "noncontiguous"])
def test_forced_torch_grouped_gemm_rejects_malformed_offsets(offset_layout):
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16)
    base = torch.tensor([32, 0, 64, 0], device="cuda", dtype=torch.int32)
    offs = base[::2]
    if offset_layout == "rank":
        offs = offs[None, :]

    with pytest.raises(KernelSelectionError, match="torch.*offsets.*(rank|contiguous)"):
        grouped_gemm(a, b, offs, backend="torch")


@cuda_only
def test_forced_torch_grouped_gemm_rejects_cross_device_offsets():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([32, 64], device="cpu", dtype=torch.int32)

    with pytest.raises(KernelSelectionError, match="torch.*same device"):
        grouped_gemm(a, b, offs, backend="torch")


@cuda_only
def test_torch_grouped_can_implement_rejects_pre_sm80(monkeypatch):

    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

    support = torch_gemm.can_implement_grouped_gemm((a, b, offs, None), {})

    assert not support.ok
    assert support.reason is not None and "SM80" in support.reason


@cuda_only
def test_forced_torch_scaled_gemm_rejects_cpu_bias():
    aq = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
    bq = torch.randn(64, 48, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(64, 1, device="cuda")
    sb = torch.ones(1, 48, device="cuda")
    bias = torch.ones(48, device="cpu", dtype=torch.bfloat16)

    with pytest.raises(KernelSelectionError, match="torch.*same device"):
        scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 0, bias=bias, backend="torch")
