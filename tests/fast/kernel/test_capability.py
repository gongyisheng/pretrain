import pytest

from src.kernel.spec import (
    CPU,
    BACKENDS,
    KernelSpec,
    PlatformInfo,
    cuda,
)

SM89 = PlatformInfo("cuda", (8, 9))
SM100 = PlatformInfo("cuda", (10, 0))
SM120 = PlatformInfo("cuda", (12, 0))
HOST = PlatformInfo()


def _fn():
    return None


def _spec(capabilities):
    return KernelSpec(
        op="test.op",
        backend="triton",
        fn=_fn,
        build="jit",
        autograd=False,
        capabilities=capabilities,
    )


def test_cpu_requirement_rejects_cuda_platform():
    assert CPU.is_satisfied_by(HOST)
    assert not CPU.is_satisfied_by(SM120)


def test_cuda_requirement_without_window_accepts_any_arch():
    assert cuda().is_satisfied_by(SM89)
    assert cuda().is_satisfied_by(SM120)
    assert not cuda().is_satisfied_by(HOST)


@pytest.mark.parametrize(
    "platform,expected",
    [
        (SM89, False),
        (SM100, True),
        (PlatformInfo("cuda", (11, 0)), True),
        (SM120, False),
    ],
)
def test_min_and_max_arch_bound_the_window_inclusively(platform, expected):
    requirement = cuda(min_arch=(10, 0), max_arch=(11, 0))

    assert requirement.is_satisfied_by(platform) is expected


def test_min_arch_alone_is_a_floor():
    assert not cuda(min_arch=(8, 9)).is_satisfied_by(PlatformInfo("cuda", (8, 0)))
    assert cuda(min_arch=(8, 9)).is_satisfied_by(SM89)


def test_arch_window_rejects_a_platform_with_no_arch():
    assert not cuda(min_arch=(8, 0)).is_satisfied_by(PlatformInfo("cuda", None))


def test_empty_capabilities_run_anywhere():
    spec = _spec(frozenset())

    assert spec.is_available(HOST)
    assert spec.is_available(SM120)


def test_capabilities_are_or_folded():
    spec = _spec(frozenset({CPU, cuda(min_arch=(10, 0))}))

    assert spec.is_available(HOST)
    assert spec.is_available(SM100)
    assert not spec.is_available(SM89)


def test_detect_reports_cpu_when_cuda_is_absent(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert PlatformInfo.detect() == PlatformInfo("cpu", None)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown kernel backend"):
        KernelSpec(op="test.op", backend="cutlass", fn=_fn, build="jit", autograd=False)


def test_non_bool_autograd_is_rejected():
    with pytest.raises(TypeError, match="autograd must be bool, got int"):
        KernelSpec(op="test.op", backend="triton", fn=_fn, build="jit", autograd=1)


def test_backends_set_is_exactly_the_three_supported_names():
    assert BACKENDS == frozenset({"eager", "triton", "cublaslt"})
