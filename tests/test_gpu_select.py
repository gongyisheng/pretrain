"""Unit tests for src/utils/gpu_select.py."""

import pytest

from src.utils.gpu_select import GpuInfo, parse_nvidia_smi, pick_available


# --- parse_nvidia_smi ---


def test_parse_single_gpu():
    out = "0, 5, 24000\n"
    assert parse_nvidia_smi(out) == [GpuInfo(index=0, util_pct=5, free_mib=24000)]


def test_parse_multiple_gpus():
    out = "0, 5, 24000\n1, 80, 1000\n2, 0, 30000\n"
    assert parse_nvidia_smi(out) == [
        GpuInfo(index=0, util_pct=5, free_mib=24000),
        GpuInfo(index=1, util_pct=80, free_mib=1000),
        GpuInfo(index=2, util_pct=0, free_mib=30000),
    ]


def test_parse_handles_extra_whitespace():
    out = "  0 ,  5 ,  24000  \n"
    assert parse_nvidia_smi(out) == [GpuInfo(index=0, util_pct=5, free_mib=24000)]


def test_parse_ignores_blank_lines():
    out = "\n0, 5, 24000\n\n"
    assert parse_nvidia_smi(out) == [GpuInfo(index=0, util_pct=5, free_mib=24000)]


def test_parse_empty_output():
    assert parse_nvidia_smi("") == []


def test_parse_raises_on_wrong_field_count():
    with pytest.raises(ValueError, match="3 comma-separated fields"):
        parse_nvidia_smi("0, 5\n")


# --- pick_available ---


def test_pick_returns_first_available():
    gpus = [
        GpuInfo(index=0, util_pct=5, free_mib=24000),
        GpuInfo(index=1, util_pct=0, free_mib=30000),
    ]
    assert pick_available(gpus) == 0


def test_pick_rejects_high_util():
    gpus = [
        GpuInfo(index=0, util_pct=11, free_mib=24000),
        GpuInfo(index=1, util_pct=5, free_mib=24000),
    ]
    assert pick_available(gpus) == 1


def test_pick_rejects_low_memory():
    gpus = [
        GpuInfo(index=0, util_pct=5, free_mib=8191),
        GpuInfo(index=1, util_pct=5, free_mib=8192),
    ]
    assert pick_available(gpus) == 1


def test_pick_returns_none_when_all_busy():
    gpus = [
        GpuInfo(index=0, util_pct=50, free_mib=24000),
        GpuInfo(index=1, util_pct=5, free_mib=4000),
    ]
    assert pick_available(gpus) is None


def test_pick_returns_none_for_empty_list():
    assert pick_available([]) is None


def test_pick_threshold_boundaries():
    # exactly at threshold = available
    gpus = [GpuInfo(index=0, util_pct=10, free_mib=8192)]
    assert pick_available(gpus) == 0


def test_pick_custom_thresholds_accepts():
    gpus = [GpuInfo(index=0, util_pct=30, free_mib=4096)]
    assert pick_available(gpus, max_util_pct=50, min_free_mib=2048) == 0


def test_pick_custom_thresholds_rejects():
    gpus = [GpuInfo(index=0, util_pct=30, free_mib=4096)]
    assert pick_available(gpus, max_util_pct=20, min_free_mib=2048) is None
