import pytest

from src.kernel.spec import (
    CPU,
    KernelSpec,
    PlatformInfo,
    cuda,
)

SM80 = PlatformInfo("cuda", (8, 0))
SM89 = PlatformInfo("cuda", (8, 9))
SM100 = PlatformInfo("cuda", (10, 0))
SM110 = PlatformInfo("cuda", (11, 0))
SM120 = PlatformInfo("cuda", (12, 0))
NO_ARCH = PlatformInfo("cuda", None)
HOST = PlatformInfo()


def _fn():
    return None


def _spec(**overrides):
    kwargs = dict(op="test.op", backend="triton", fn=_fn, build="jit", autograd=False)
    return KernelSpec(**(kwargs | overrides))


@pytest.mark.parametrize(
    "requirement,platform,expected",
    [
        (CPU, HOST, True),
        (CPU, SM120, False),
        (cuda(), SM89, True),
        (cuda(), SM120, True),
        (cuda(), HOST, False),
        (cuda((10, 0), (11, 0)), SM89, False),
        (cuda((10, 0), (11, 0)), SM100, True),
        (cuda((10, 0), (11, 0)), SM110, True),
        (cuda((10, 0), (11, 0)), SM120, False),
        (cuda(min_arch=(8, 9)), SM80, False),
        (cuda(min_arch=(8, 9)), SM89, True),
        (cuda(min_arch=(8, 0)), NO_ARCH, False),
    ],
)
def test_capability_requirement_is_satisfied_by(requirement, platform, expected):
    assert requirement.is_satisfied_by(platform) is expected


@pytest.mark.parametrize(
    "capabilities,platform,expected",
    [
        (frozenset(), HOST, True),
        (frozenset(), SM120, True),
        (frozenset({CPU, cuda(min_arch=(10, 0))}), HOST, True),
        (frozenset({CPU, cuda(min_arch=(10, 0))}), SM100, True),
        (frozenset({CPU, cuda(min_arch=(10, 0))}), SM89, False),
    ],
)
def test_kernel_spec_is_available(capabilities, platform, expected):
    assert _spec(capabilities=capabilities).is_available(platform) is expected


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"autograd": 1}, TypeError),
        ({"autograd": None}, TypeError),
        ({"backend": "unknown"}, ValueError),
    ],
)
def test_kernel_spec_post_init_raise_error(overrides, error):
    with pytest.raises(error):
        _spec(**overrides)


@pytest.mark.parametrize(
    "cuda_available,capability,expected",
    [
        (False, None, HOST),
        (True, (8, 9), SM89),
        (True, (12, 0), SM120),
    ],
)
def test_platform_info_detect(monkeypatch, cuda_available, capability, expected):
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_available)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: capability)

    assert PlatformInfo.detect() == expected
