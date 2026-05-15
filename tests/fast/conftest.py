"""Shared fixtures for tests/fast.

Sets the default torch device for the whole `tests/fast` session.
Selection order:
  1. `--device=<cpu|cuda>` pytest CLI flag
  2. cuda if torch.cuda.is_available() else cpu
"""
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run tests/fast on. Default: cuda if available, else cpu.",
    )


def _select_device(config) -> str:
    chosen = config.getoption("--device")
    if chosen is not None:
        if chosen == "cuda" and not torch.cuda.is_available():
            pytest.exit("--device=cuda requested but torch.cuda.is_available() is False", returncode=1)
        return chosen
    return "cuda" if torch.cuda.is_available() else "cpu"


# Tests that manage their own device internally and must NOT have the global
# default device overridden. The Trainer constructs a DataLoader, which creates
# an internal CPU torch.Generator; with set_default_device("cuda"), DataLoader's
# `tensor.random_(generator=cpu_gen)` raises:
#   "Expected a 'cuda' device type for generator but found 'cpu'"
# These tests already pick the right device via Trainer.__init__'s own check.
_OPT_OUT_PATHS = ("training/test_trainer",)


@pytest.fixture(autouse=True)
def _set_default_device(request):
    if any(p in request.node.nodeid for p in _OPT_OUT_PATHS):
        yield
        return
    device = _select_device(request.config)
    prev = torch.get_default_device()
    torch.set_default_device(device)
    yield device
    torch.set_default_device(prev)


@pytest.fixture(scope="session")
def device(request) -> str:
    return _select_device(request.config)
