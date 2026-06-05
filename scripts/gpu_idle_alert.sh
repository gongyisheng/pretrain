#!/bin/bash
# Monitor GPU utilization and email when the GPU goes idle. Intended as a cronjob.
#
# Idle = every GPU stays at/below UTIL_THRESHOLD across all samples (sustained,
# so brief between-step dips during training don't trigger a false alarm).
#
# To avoid spamming on every cron tick, a state file debounces: an email is sent
# once on the busy -> idle transition and re-armed once the GPU is busy again.
#
# Config via env vars (defaults in parens):
#   GPU_IDLE_UTIL_THRESHOLD  utilization % at/below which a GPU counts as idle (5)
#   GPU_IDLE_SAMPLES         number of samples that must all be idle (6)
#   GPU_IDLE_INTERVAL        seconds between samples (5)
#   GPU_IDLE_STATE_FILE      debounce state file (/tmp/gpu_idle_alert.state)
#   GPU_IDLE_NOTIFY_BUSY     if "1", also email when GPU returns to busy (0)
#
# Cron example (check every 10 min, log output):
#   */10 * * * * /home/yisheng/Documents/pretrain/scripts/gpu_idle_alert.sh >> /tmp/gpu_idle_alert.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

UTIL_THRESHOLD=${GPU_IDLE_UTIL_THRESHOLD:-5}
SAMPLES=${GPU_IDLE_SAMPLES:-6}
INTERVAL=${GPU_IDLE_INTERVAL:-5}
STATE_FILE=${GPU_IDLE_STATE_FILE:-/tmp/gpu_idle_alert.state}
NOTIFY_BUSY=${GPU_IDLE_NOTIFY_BUSY:-0}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found; aborting."
    exit 1
fi

# Sample max utilization across all GPUs, SAMPLES times. Idle only if every
# sample's busiest GPU is at/below the threshold.
idle=1
max_util=0
for ((i = 0; i < SAMPLES; i++)); do
    if [ "$i" -gt 0 ]; then sleep "$INTERVAL"; fi
    # Highest utilization among all GPUs for this sample.
    sample_max=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
        | awk 'BEGIN{m=0} {v=$1+0; if (v>m) m=v} END{print m}')
    if [ "$sample_max" -gt "$max_util" ]; then max_util=$sample_max; fi
    if [ "$sample_max" -gt "$UTIL_THRESHOLD" ]; then
        idle=0
        break
    fi
done

prev_state="busy"
if [ -f "$STATE_FILE" ]; then prev_state=$(cat "$STATE_FILE"); fi

send_email() {
    # Prefer the project's uv env; fall back to plain python3.
    if command -v uv >/dev/null 2>&1; then
        (cd "$REPO_DIR" && uv run python scripts/send_email.py --subject "$1" --body "$2")
    else
        (cd "$REPO_DIR" && python3 scripts/send_email.py --subject "$1" --body "$2")
    fi
}

if [ "$idle" -eq 1 ]; then
    log "GPU idle (max util ${max_util}% over ${SAMPLES} samples, threshold ${UTIL_THRESHOLD}%)."
    if [ "$prev_state" != "idle" ]; then
        host=$(hostname)
        body=$(printf 'Host: %s\nMax GPU utilization: %s%%\nThreshold: %s%%\nSamples: %s @ %ss\nTime: %s\n\n%s' \
            "$host" "$max_util" "$UTIL_THRESHOLD" "$SAMPLES" "$INTERVAL" "$(date)" \
            "$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader)")
        send_email "[$host] GPU is IDLE" "$body" && log "Idle alert sent."
        echo "idle" > "$STATE_FILE"
    else
        log "Already alerted for this idle period; skipping email."
    fi
else
    log "GPU busy (max util ${max_util}% > threshold ${UTIL_THRESHOLD}%)."
    if [ "$prev_state" = "idle" ] && [ "$NOTIFY_BUSY" = "1" ]; then
        host=$(hostname)
        send_email "[$host] GPU is BUSY again" \
            "Host: $host"$'\n'"Max GPU utilization: ${max_util}%"$'\n'"Time: $(date)" \
            && log "Busy alert sent."
    fi
    echo "busy" > "$STATE_FILE"
fi
