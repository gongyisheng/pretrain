"""Unit tests for experiments/superbpe/collect_results.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    """Import collect_results.py without putting experiments/ on sys.path."""
    path = Path("experiments/superbpe/collect_results.py").resolve()
    spec = importlib.util.spec_from_file_location("collect_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_eval_log(folder: Path, name: str, bytes_per_token: float) -> None:
    """Write a fake eval-script log mirroring scripts/eval_tokenizer.py output."""
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokenizer": f"tokenizers/experiments/{name}",
        "bytes_per_token": bytes_per_token,
    }
    text = (
        f"Evaluating tokenizers/experiments/{name} on openwebtext:train (10000 docs)\n"
        f"{json.dumps(payload, indent=2)}\n"
    )
    (folder / f"{name}.log").write_text(text)


def test_parse_name_bpe():
    mod = _load_module()
    assert mod.parse_name("bpe_v200k") == {
        "method": "bpe",
        "V": 200,
        "t": None,
        "m": None,
    }


def test_parse_name_superbpe():
    mod = _load_module()
    assert mod.parse_name("superbpe_v100k_t40k_m3") == {
        "method": "superbpe",
        "V": 100,
        "t": 40,
        "m": 3,
    }


def test_extract_bytes_per_token_from_log(tmp_path: Path):
    mod = _load_module()
    _write_eval_log(tmp_path, "bpe_v50k", 4.123)
    bpt = mod.extract_bytes_per_token(tmp_path / "bpe_v50k.log")
    assert bpt == pytest.approx(4.123)


def test_collect_writes_csv_with_stable_columns(tmp_path: Path):
    mod = _load_module()
    eval_dir = tmp_path / "eval"
    _write_eval_log(eval_dir, "bpe_v50k", 4.10)
    _write_eval_log(eval_dir, "superbpe_v100k_t40k_m3", 5.21)

    out = tmp_path / "results.csv"
    rows = mod.collect(eval_dir, out)
    assert len(rows) == 2

    text = out.read_text().splitlines()
    # Header is the documented stable schema.
    assert (
        text[0]
        == "name,V,t,m,method,bytes_per_token,stage2_merges_accepted,wall_seconds"
    )
    # Rows sorted by (V, method, t, m) — bpe_v50k before superbpe_v100k_t40k_m3.
    assert text[1] == "bpe_v50k,50,,,bpe,4.1,,"
    assert text[2] == "superbpe_v100k_t40k_m3,100,40,3,superbpe,5.21,,"


def test_collect_skips_unparseable_logs(tmp_path: Path, capsys):
    mod = _load_module()
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "garbage.log").write_text("not json at all\n")
    out = tmp_path / "results.csv"
    rows = mod.collect(eval_dir, out)
    assert rows == []
    err = capsys.readouterr().err
    assert "skip" in err.lower()
