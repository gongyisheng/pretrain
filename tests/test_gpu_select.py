"""Unit tests for src/utils/gpu_select.py."""

import importlib.util
import pathlib
import sys

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


# --- CLI integration ---


def _load_cli_module():
    """Load scripts/pick_free_gpu.py as a module (scripts/ has no __init__.py)."""
    path = (
        pathlib.Path(__file__).resolve().parent.parent / "scripts" / "pick_free_gpu.py"
    )
    spec = importlib.util.spec_from_file_location("pick_free_gpu", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pick_free_gpu"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cli_prints_first_available_gpu(monkeypatch, capsys):
    cli = _load_cli_module()

    def fake_query():
        return "0, 50, 24000\n1, 5, 24000\n"

    monkeypatch.setattr(cli, "query_nvidia_smi", fake_query)
    monkeypatch.setattr(cli, "POLL_INTERVAL_S", 0)

    cli.main()
    out = capsys.readouterr().out.strip()
    assert out == "1"


def test_cli_waits_until_gpu_free(monkeypatch, capsys):
    cli = _load_cli_module()

    calls = {"n": 0}

    def fake_query():
        calls["n"] += 1
        if calls["n"] < 3:
            return "0, 80, 24000\n"  # busy
        return "0, 2, 24000\n"  # idle

    sleeps: list[float] = []
    monkeypatch.setattr(cli, "query_nvidia_smi", fake_query)
    monkeypatch.setattr(cli, "POLL_INTERVAL_S", 0)
    monkeypatch.setattr(cli.time, "sleep", lambda s: sleeps.append(s))

    cli.main()
    assert capsys.readouterr().out.strip() == "0"
    assert calls["n"] == 3
    assert len(sleeps) == 2  # slept before each of the first two retries


def test_cli_retries_on_nvidia_smi_error(monkeypatch, capsys):
    import subprocess

    cli = _load_cli_module()
    calls = {"n": 0}

    def fake_query():
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="boom")
        return "0, 0, 24000\n"

    monkeypatch.setattr(cli, "query_nvidia_smi", fake_query)
    monkeypatch.setattr(cli, "POLL_INTERVAL_S", 0)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)

    cli.main()
    assert capsys.readouterr().out.strip() == "0"
    assert calls["n"] == 2
