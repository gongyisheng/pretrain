#!/usr/bin/env python3
"""
Set power limit on NVIDIA GPUs via NVML. Requires root.

sudo apt install -y python3-pynvml
sudo python3 scripts/gpu_power.py 300           # all GPUs, 300W
sudo python3 scripts/gpu_power.py 300 --gpu 0   # GPU 0 only
sudo python3 scripts/gpu_power.py 300 --gpu 0,1 # GPUs 0 and 1
sudo python3 scripts/gpu_power.py default       # restore driver default
"""

import argparse
import sys
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerManagementLimit,
    nvmlDeviceGetPowerManagementDefaultLimit,
    nvmlDeviceGetPowerManagementLimitConstraints,
    nvmlDeviceSetPowerManagementLimit,
    NVMLError,
)


def parse_watts(arg: str) -> int | None:
    arg = arg.lower()
    if arg == "default":
        return None
    value = int(arg)
    if value <= 0:
        raise ValueError("power limit must be a positive integer (watts)")
    return value


def parse_gpus(arg: str | None, total: int) -> list[int]:
    if arg is None:
        return list(range(total))
    indices = [int(x) for x in arg.split(",") if x.strip()]
    for i in indices:
        if not 0 <= i < total:
            raise ValueError(f"GPU index {i} out of range (have {total} GPUs)")
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Set NVIDIA GPU power limit via NVML.")
    parser.add_argument(
        "watts",
        help="power limit in watts (e.g. 300), or 'default' to restore"
    )
    parser.add_argument(
        "--gpu",
        help="comma-separated GPU indices (e.g. 0 or 0,1). Defaults to all GPUs."
    )
    args = parser.parse_args()

    try:
        target_w = parse_watts(args.watts)
    except ValueError as e:
        parser.error(str(e))

    nvmlInit()
    try:
        gpus = parse_gpus(args.gpu, nvmlDeviceGetCount())
        for i in gpus:
            h = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(h)
            before_mw = nvmlDeviceGetPowerManagementLimit(h)
            default_mw = nvmlDeviceGetPowerManagementDefaultLimit(h)
            min_mw, max_mw = nvmlDeviceGetPowerManagementLimitConstraints(h)

            apply_mw = default_mw if target_w is None else target_w * 1000
            tag = "default" if target_w is None else f"{target_w}W"

            if apply_mw < min_mw or apply_mw > max_mw:
                print(
                    f"GPU {i} ({name}): {target_w}W out of range "
                    f"[{min_mw // 1000}W, {max_mw // 1000}W]",
                    file=sys.stderr,
                )
                continue

            try:
                nvmlDeviceSetPowerManagementLimit(h, apply_mw)
            except NVMLError as e:
                print(f"GPU {i} ({name}): FAILED -- {e}", file=sys.stderr)
                continue

            after_mw = nvmlDeviceGetPowerManagementLimit(h)
            print(
                f"GPU {i} ({name}): {before_mw // 1000}W -> {after_mw // 1000}W (target {tag})"
            )
    finally:
        nvmlShutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
