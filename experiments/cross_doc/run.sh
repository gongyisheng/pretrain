#!/usr/bin/env bash
set -euo pipefail

echo "=== Run 1: baseline (no doc_mask) ==="
uv run python scripts/train.py --config experiments/cross_doc/configs/baseline.yaml

echo "=== Run 2: doc_mask enabled ==="
uv run python scripts/train.py --config experiments/cross_doc/configs/doc_mask.yaml

echo "=== Done. Compare runs in W&B under group: cross-doc-sweep ==="
