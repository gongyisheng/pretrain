#!/bin/bash
# Install a systemd service that applies 400 W now and at every boot.
# Usage: sudo ./scripts/nvidia_hardware_daemon.sh [GPU index or UUID]
# Defaults to GPU 0; the service stores its UUID to survive index changes.
# To stop applying the limit at boot:
#   sudo systemctl disable --now nvidia_hardware_daemon.service
# Disabling the service leaves the current power limit in place.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: sudo $0 [GPU index or UUID]"
    echo "Set GPU 0 (or the selected GPU) to 400 W now and at every boot."
    exit 0
fi

if (( $# > 1 )); then
    echo "Usage: sudo $0 [GPU index or UUID]" >&2
    exit 1
fi

if (( EUID != 0 )); then
    echo "Run this script with sudo to install the systemd service." >&2
    exit 1
fi

if [[ ! -d /run/systemd/system || ! -x /usr/bin/nvidia-smi ]]; then
    echo "This script requires systemd and /usr/bin/nvidia-smi." >&2
    exit 1
fi

GPU_UUID=$(/usr/bin/nvidia-smi -i "${1:-0}" --query-gpu=uuid --format=csv,noheader)
if [[ ! $GPU_UUID =~ ^GPU-[[:xdigit:]-]+$ ]]; then
    echo "Select exactly one GPU by index or UUID." >&2
    exit 1
fi

POWER_RANGE=$(/usr/bin/nvidia-smi -i "$GPU_UUID" \
    --query-gpu=power.min_limit,power.max_limit --format=csv,noheader,nounits)
if ! awk -F, 'NF == 2 && $1 + 0 > 0 && $1 + 0 <= 400 && $2 + 0 >= 400 { valid = 1 }
    END { exit !valid }' <<< "$POWER_RANGE"; then
    echo "GPU does not report support for 400 W (min, max: $POWER_RANGE)." >&2
    exit 1
fi

cat > /etc/systemd/system/nvidia_hardware_daemon.service <<EOF
[Unit]
Description=Set NVIDIA GPU power limit to 400 W
After=nvidia-persistenced.service
StartLimitIntervalSec=0

[Service]
Type=oneshot
ExecStartPre=/usr/bin/nvidia-smi -i $GPU_UUID -pm 1
ExecStart=/usr/bin/nvidia-smi -i $GPU_UUID -pl 400
RemainAfterExit=yes
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
chmod 644 /etc/systemd/system/nvidia_hardware_daemon.service

systemctl daemon-reload
systemctl enable nvidia_hardware_daemon.service
# Restart also applies changes when an earlier version is already active.
systemctl restart nvidia_hardware_daemon.service
systemctl is-enabled nvidia_hardware_daemon.service
systemctl is-active nvidia_hardware_daemon.service
/usr/bin/nvidia-smi -i "$GPU_UUID" --query-gpu=name,power.limit --format=csv
echo "400 W boot service installed for $GPU_UUID."
echo "After reboot, verify with: nvidia-smi --query-gpu=name,power.limit --format=csv"
