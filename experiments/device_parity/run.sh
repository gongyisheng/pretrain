#!/bin/bash
# Run device-parity benchmark and emit results/<gpu>.json with the final val metrics.
# Usage: nohup bash experiments/device_parity/run.sh > logs/device_parity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

CONFIG="experiments/device_parity/qwen3_57m.yaml"
RESULTS_DIR="experiments/device_parity/results"
mkdir -p "${RESULTS_DIR}"

GPU_RAW=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_SLUG=$(echo "${GPU_RAW}" | sed -e 's/^NVIDIA //' -e 's/[^A-Za-z0-9]/_/g' -e 's/__*/_/g' -e 's/^_//' -e 's/_$//')
HOSTNAME_VAL=$(hostname)
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

TRAIN_LOG=$(mktemp)
trap 'rm -f "${TRAIN_LOG}"' EXIT

echo "=== device_parity / qwen3_57m on ${GPU_RAW} (${HOSTNAME_VAL}) ==="
echo "Started at: $(date -Iseconds)"
START_EPOCH=$(date +%s)

uv run python scripts/train.py --config "${CONFIG}" 2>&1 | tee "${TRAIN_LOG}"

END_EPOCH=$(date +%s)
WALLCLOCK=$((END_EPOCH - START_EPOCH))
echo "Finished at: $(date -Iseconds) (wallclock ${WALLCLOCK}s)"

# Parse the last "[eval] val_loss=X | val_ppl=Y" line emitted by the trainer.
LAST_EVAL=$(grep "\[eval\]" "${TRAIN_LOG}" | tail -n1 || true)
VAL_LOSS=$(echo "${LAST_EVAL}" | sed -n 's/.*val_loss=\([0-9.]*\).*/\1/p')
VAL_PPL=$(echo "${LAST_EVAL}" | sed -n 's/.*val_ppl=\([0-9.]*\).*/\1/p')

OUT="${RESULTS_DIR}/${GPU_SLUG}.json"
cat > "${OUT}" <<EOF
{
  "gpu": "${GPU_RAW}",
  "hostname": "${HOSTNAME_VAL}",
  "commit": "${COMMIT}",
  "config": "${CONFIG}",
  "val_loss": ${VAL_LOSS:-null},
  "val_ppl": ${VAL_PPL:-null},
  "wallclock_sec": ${WALLCLOCK},
  "finished_at": "$(date -Iseconds)"
}
EOF

echo "Wrote ${OUT}"
