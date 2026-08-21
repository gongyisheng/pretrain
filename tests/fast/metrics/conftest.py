"""Shared fixtures for tests/fast/metrics/ — disarms the module-global
monitoring flags after every test."""

import pytest

from src.metrics.activation import set_activation_monitoring_status
from src.metrics.quant import set_quantization_monitoring_status


@pytest.fixture(autouse=True)
def _disarm_monitoring():
    """Both _RECORDING flags are module-global: without this, one failing test
    leaves monitoring armed for every later test in the same xdist worker."""
    yield
    set_activation_monitoring_status(False)
    set_quantization_monitoring_status(False)
