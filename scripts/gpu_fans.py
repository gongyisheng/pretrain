#!/usr/bin/env python3
"""
Set fan speed on all NVIDIA GPUs via NVML. Requires root.

sudo apt install -y python3-pynvml
sudo python3 scripts/gpu_fans.py 70
"""
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


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: gpu_fans.py <0-100|auto>", file=sys.stderr)
        return 2
    arg = sys.argv[1].lower()
    target: int | None
    if arg == "auto":
        target = None
    else:
        try:
            target = int(arg)
        except ValueError:
            print(f"invalid speed: {arg}", file=sys.stderr)
            return 2
        if not 0 <= target <= 100:
            print("speed must be 0-100", file=sys.stderr)
            return 2

    nvmlInit()
    try:
        for i in range(nvmlDeviceGetCount()):
            h = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(h)
            n_fans = nvmlDeviceGetNumFans(h)
            for f in range(n_fans):
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
