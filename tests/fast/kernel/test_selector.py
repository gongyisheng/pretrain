import pytest
import torch

from src.kernel import selector
from src.kernel.registry import KERNEL_REGISTRY, KernelRegistry
from src.kernel.selector import KernelSelectionError, dispatch, select_kernel
from src.kernel.spec import CPU, KernelSpec, PlatformInfo, cuda


def _spec(backend, capabilities=frozenset(), *, op="test.op", fn=None, reference=False):
    return KernelSpec(
        op=op,
        backend=backend,
        fn=fn or (lambda *a, **k: backend),
        build="jit",
        autograd=False,
        capabilities=capabilities,
        reference=reference,
    )


def _registry(*specs):
    registry = KernelRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


def _register_for_dispatch_ops(monkeypatch):
    """dispatch reads the global registry and the live CUDA arch; pin both."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: (8, 9))
    for op, backends in (
        (ADD_OP, [ADD_ON_CPU]),
        (SPLIT_OP, [ON_CPU, ON_CUDA_80]),
        (CUDA_ONLY_OP, [ON_CUDA_80, ON_CUDA_100]),
    ):
        done = {spec.backend for spec in KERNEL_REGISTRY.implementations(op)}
        for backend, capabilities, fn in backends:
            if backend not in done:
                KERNEL_REGISTRY.register(_spec(backend, capabilities, op=op, fn=fn))


@pytest.fixture(autouse=True)
def _clear_caches():
    selector.clear_cache()
    yield
    selector.clear_cache()


SM89 = PlatformInfo("cuda", (8, 9))
SM100 = PlatformInfo("cuda", (10, 0))
HOST = PlatformInfo()

EAGER_CPU = _spec("eager", frozenset({CPU}))
EAGER_REF = _spec("eager", reference=True)
TRITON_REF = _spec("triton", reference=True)
TRITON_MIN80 = _spec("triton", frozenset({cuda((8, 0))}))
TRITON_MIN89 = _spec("triton", frozenset({cuda((8, 9))}))
TRITON_80_TO_85 = _spec("triton", frozenset({cuda((8, 0), (8, 5))}))
CUBLASLT_MIN99 = _spec("cublaslt", frozenset({cuda((99, 0))}))
CUBLASLT_100_TO_110 = _spec("cublaslt", frozenset({cuda((10, 0), (11, 0))}))


SELECT_KERNEL_CASES = [
    # An explicit backend and a sole registration both skip every later gate.
    ([CUBLASLT_MIN99], "cublaslt", None, "cublaslt"),
    ([CUBLASLT_MIN99], None, None, "cublaslt"),
    ([EAGER_CPU, TRITON_MIN80], None, HOST, "eager"),
    ([EAGER_REF, TRITON_MIN80], None, SM89, "triton"),
    ([EAGER_REF, CUBLASLT_100_TO_110], None, SM89, "eager"),
    ([EAGER_REF, TRITON_MIN80], None, HOST, "eager"),
]

SELECT_KERNEL_ERROR_CASES = [
    ([], None, None),
    ([EAGER_CPU, TRITON_MIN80], "cublaslt", None),
    ([EAGER_CPU, TRITON_MIN80], None, None),
    # No reference spec here on purpose: a reference is always eligible (empty
    # capabilities), so it would never let this scenario -- nothing eligible at
    # all -- occur.
    ([TRITON_80_TO_85, CUBLASLT_100_TO_110], None, SM89),
    ([TRITON_MIN89, CUBLASLT_100_TO_110], None, SM100),
    ([EAGER_REF, TRITON_MIN89, CUBLASLT_100_TO_110], None, SM100),
    ([EAGER_REF, TRITON_REF], None, HOST),
]


CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda", 0)

# (backend, capabilities, fn); a None fn returns the backend name, so a dispatch
# result of "triton" means the triton implementation is the one that ran.
ADD_ON_CPU = ("eager", frozenset({CPU}), lambda x, y: x + y)
ON_CPU = ("eager", frozenset({CPU}), None)
ON_CUDA_80 = ("triton", frozenset({cuda((8, 0))}), None)
ON_CUDA_100 = ("cublaslt", frozenset({cuda((10, 0))}), None)

ADD_OP = "d.add"
SPLIT_OP = "d.split"
CUDA_ONLY_OP = "d.cuda_only"
MISSING_OP = "d.missing"

# Regression guard on "on-cpu": the platform gate comes from the call's own device, not
# from a process-global fact. Resolving by process-global platform used to route CPU
# calls into the CUDA-only triton backend and crash.
DISPATCH_CASES = [
    (ADD_OP, None, CPU_DEVICE, (2, 3), {}, 5),
    (ADD_OP, None, CPU_DEVICE, (2,), {"y": 3}, 5),
    (SPLIT_OP, None, CPU_DEVICE, (), {}, "eager"),
    (SPLIT_OP, None, CUDA_DEVICE, (), {}, "triton"),
    (SPLIT_OP, "triton", CPU_DEVICE, (), {}, "triton"),
]

DISPATCH_ERROR_CASES = [
    (MISSING_OP, None),
    (SPLIT_OP, "cublaslt"),
    (CUDA_ONLY_OP, None),
]


@pytest.mark.parametrize("specs,backend,platform,expected", SELECT_KERNEL_CASES)
def test_select_kernel(specs, backend, platform, expected):
    spec = select_kernel("test.op", backend, platform, _registry(*specs))

    assert spec.backend == expected


@pytest.mark.parametrize("specs,backend,platform", SELECT_KERNEL_ERROR_CASES)
def test_select_kernel_raise_error(specs, backend, platform):
    with pytest.raises(KernelSelectionError):
        select_kernel("test.op", backend, platform, _registry(*specs))


@pytest.mark.parametrize("op,backend,device,args,kwargs,expected", DISPATCH_CASES)
def test_dispatch_pass(monkeypatch, op, backend, device, args, kwargs, expected):
    _register_for_dispatch_ops(monkeypatch)

    assert dispatch(op, args, kwargs, backend, device=device) == expected


@pytest.mark.parametrize("op,backend", DISPATCH_ERROR_CASES)
def test_dispatch_raise_error(monkeypatch, op, backend):
    _register_for_dispatch_ops(monkeypatch)

    with pytest.raises(KernelSelectionError):
        dispatch(op, (), {}, backend, device=CPU_DEVICE)
