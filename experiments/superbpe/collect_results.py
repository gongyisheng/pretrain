"""Aggregate per-run eval logs into experiments/superbpe/results.csv.

Reads JSON blobs printed by scripts/eval_tokenizer.py from
logs/superbpe/eval/*.log and writes a CSV with the stable schema:

    name,V,t,m,method,bytes_per_token,stage2_merges_accepted,wall_seconds

`stage2_merges_accepted` and `wall_seconds` are intentionally blank in
this iteration — they require parsing train logs. The columns are
emitted so the schema does not change when those values land later.

Usage:

    uv run python experiments/superbpe/collect_results.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

EVAL_LOG_DIR = Path("logs/superbpe/eval")
RESULTS_CSV = Path("experiments/superbpe/results.csv")

COLUMNS = [
    "name",
    "V",
    "t",
    "m",
    "method",
    "bytes_per_token",
    "stage2_merges_accepted",
    "wall_seconds",
]


def parse_name(stem: str) -> dict:
    """Parse 'bpe_v200k' / 'superbpe_v200k_t80k_m4' into a fields dict."""
    parts = stem.split("_")
    if (
        parts[0] == "bpe"
        and len(parts) == 2
        and parts[1].startswith("v")
        and parts[1].endswith("k")
    ):
        return {"method": "bpe", "V": int(parts[1][1:-1]), "t": None, "m": None}
    if (
        parts[0] == "superbpe"
        and len(parts) == 4
        and parts[1].startswith("v")
        and parts[1].endswith("k")
        and parts[2].startswith("t")
        and parts[2].endswith("k")
        and parts[3].startswith("m")
    ):
        return {
            "method": "superbpe",
            "V": int(parts[1][1:-1]),
            "t": int(parts[2][1:-1]),
            "m": int(parts[3][1:]),
        }
    raise ValueError(f"unrecognized name: {stem}")


_JSON_RE = re.compile(r"\{[^{}]*\"bytes_per_token\"[^{}]*\}", re.DOTALL)


def extract_bytes_per_token(log_path: Path) -> float:
    """Extract bytes_per_token from a single eval log."""
    text = log_path.read_text()
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON blob with bytes_per_token in {log_path}")
    payload = json.loads(match.group(0))
    return float(payload["bytes_per_token"])


def collect(eval_dir: Path, output_csv: Path) -> list[dict]:
    """Walk eval_dir, parse logs, write CSV; return the list of row dicts."""
    rows: list[dict] = []
    for log in sorted(eval_dir.glob("*.log")):
        try:
            fields = parse_name(log.stem)
            bpt = extract_bytes_per_token(log)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"skip {log.name}: {exc}", file=sys.stderr)
            continue
        rows.append(
            {
                "name": log.stem,
                "V": fields["V"],
                "t": fields["t"] if fields["t"] is not None else "",
                "m": fields["m"] if fields["m"] is not None else "",
                "method": fields["method"],
                "bytes_per_token": bpt,
                "stage2_merges_accepted": "",
                "wall_seconds": "",
            }
        )

    # Sort by (V, method, t, m) — bpe within a V comes before superbpe (since "bpe" < "superbpe").
    rows.sort(
        key=lambda r: (
            r["V"],
            r["method"],
            r["t"] if r["t"] != "" else -1,
            r["m"] if r["m"] != "" else -1,
        )
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    if not EVAL_LOG_DIR.exists():
        print(
            f"error: {EVAL_LOG_DIR} not found — run experiments/superbpe/run_eval.sh first",
            file=sys.stderr,
        )
        return 1
    rows = collect(EVAL_LOG_DIR, RESULTS_CSV)
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
