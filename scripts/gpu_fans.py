#!/usr/bin/env python3
"""
Set fan speed on NVIDIA GPUs via NVML. Requires root.

sudo apt install -y python3-pynvml
sudo python3 scripts/gpu_fans.py 70           # all GPUs
sudo python3 scripts/gpu_fans.py 70 --gpu 0   # GPU 0 only
sudo python3 scripts/gpu_fans.py 70 --gpu 0,1 # GPUs 0 and 1
sudo python3 scripts/gpu_fans.py auto         # hand control back to driver
"""

import argparse
import sys
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetNumFans,
    nvmlDeviceGetFanSpeed_v2,
    nvmlDeviceSetFanSpeed_v2,
    nvmlDeviceSetDefaultFanSpeed_v2,
    NVMLError,
)


def parse_speed(arg: str) -> int | None:
    arg = arg.lower()
    if arg == "auto":
        return None
    value = int(arg)
    if not 0 <= value <= 100:
        raise ValueError("speed must be 0-100")
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
    parser = argparse.ArgumentParser(description="Set NVIDIA GPU fan speed via NVML.")
    parser.add_argument(
        "speed",
        help="0-100 percent, or 'auto' to restore driver control"
    )
    parser.add_argument(
        "--gpu",
        help="comma-separated GPU indices (e.g. 0 or 0,1). Defaults to all GPUs.",
    )
    args = parser.parse_args()

    try:
        target = parse_speed(args.speed)
    except ValueError as e:
        parser.error(str(e))

    nvmlInit()
    try:
        gpus = parse_gpus(args.gpu, nvmlDeviceGetCount())
        for i in gpus:
            h = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(h)
            for f in range(nvmlDeviceGetNumFans(h)):
                before = nvmlDeviceGetFanSpeed_v2(h, f)
                try:
                    if target is None:
                        nvmlDeviceSetDefaultFanSpeed_v2(h, f)
                    else:
                        nvmlDeviceSetFanSpeed_v2(h, f, target)
                except NVMLError as e:
                    print(f"GPU {i} fan {f} ({name}): FAILED -- {e}", file=sys.stderr)
                    continue
                after = nvmlDeviceGetFanSpeed_v2(h, f)
                tag = "auto" if target is None else f"{target}%"
                print(f"GPU {i} fan {f} ({name}): {before}% -> {after}% (target {tag})")
    finally:
        nvmlShutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
