#!/bin/bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: sudo $0 [GPU index or UUID]"
    echo "Cap GPU 0 (or the selected GPU) at 400 W when its limit exceeds 400 W."
    echo "Check now and every 5 minutes."
    exit 0
fi

if (( $# > 1 )); then
    echo "Usage: sudo $0 [GPU index or UUID]" >&2
    exit 1
fi

if (( EUID != 0 )); then
    echo "Run this script with sudo to install the systemd service and timer." >&2
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

cat > /etc/systemd/system/nvidia_hardware_daemon.service <<EOF
[Unit]
Description=Cap NVIDIA GPU power limit at 400 W when needed
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'set -euo pipefail; export LC_ALL=C; power_limit=\$\$(/usr/bin/nvidia-smi -i $GPU_UUID --query-gpu=power.limit --format=csv,noheader,nounits); if [[ ! \$\$power_limit =~ ^[0-9]+([.][0-9]+)?\$\$ ]]; then echo "Invalid NVIDIA power limit: \$\$power_limit" >&2; exit 1; fi; if /usr/bin/awk -v power_limit="\$\$power_limit" "BEGIN { exit !(power_limit > 400) }"; then /usr/bin/nvidia-smi -i $GPU_UUID -pm 1; /usr/bin/nvidia-smi -i $GPU_UUID -pl 400; fi'

EOF

cat > /etc/systemd/system/nvidia_hardware_daemon.timer <<EOF
[Unit]
Description=Check NVIDIA GPU power limit every 5 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
Unit=nvidia_hardware_daemon.service

[Install]
WantedBy=timers.target
EOF
chmod 644 /etc/systemd/system/nvidia_hardware_daemon.service \
    /etc/systemd/system/nvidia_hardware_daemon.timer

systemctl daemon-reload
systemctl disable --now nvidia_hardware_daemon.service >/dev/null 2>&1 || true
systemctl start nvidia_hardware_daemon.service
systemctl enable --now nvidia_hardware_daemon.timer
systemctl is-enabled nvidia_hardware_daemon.timer
systemctl is-active nvidia_hardware_daemon.timer
/usr/bin/nvidia-smi -i "$GPU_UUID" --query-gpu=name,power.limit --format=csv
echo "nvidia hardware daemon timer installed for $GPU_UUID"
