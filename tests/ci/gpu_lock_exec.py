#!/usr/bin/env python3
"""Pick an idle GPU index for CI tests and print it to stdout.

Polls `nvidia-smi` until a GPU has utilization <= 10% AND free memory
>= 8 GiB, then prints the chosen index on a single line. Waits
indefinitely; the workflow's `timeout-minutes` bounds the total wait.

stderr carries per-poll status. Only the chosen integer goes to stdout
so the workflow can capture it into a step output.
"""

import subprocess
import sys
import time
from dataclasses import dataclass

POLL_INTERVAL_S = 30


@dataclass(frozen=True)
class GpuInfo:
    index: int
    util_pct: int
    free_mib: int


def parse_nvidia_smi(output: str) -> list[GpuInfo]:
    """Parse CSV body of `nvidia-smi --query-gpu=index,utilization.gpu,memory.free`."""
    gpus: list[GpuInfo] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Expected 3 comma-separated fields, got {len(parts)}: {line!r}"
            )
        idx, util, free = parts
        gpus.append(GpuInfo(index=int(idx), util_pct=int(util), free_mib=int(free)))
    return gpus


def pick_available(
    gpus: list[GpuInfo],
    max_util_pct: int = 10,
    min_free_mib: int = 8 * 1024,  # 8 GiB
) -> int | None:
    """Return the lowest-index GPU whose util and free memory clear thresholds."""
    for g in gpus:
        if g.util_pct <= max_util_pct and g.free_mib >= min_free_mib:
            return g.index
    return None


def query_nvidia_smi() -> str:
    """Return the CSV body of `nvidia-smi --query-gpu=...`.

    Raises subprocess.CalledProcessError on non-zero exit.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    while True:
        try:
            output = query_nvidia_smi()
        except subprocess.CalledProcessError as e:
            print(
                f"nvidia-smi failed (rc={e.returncode}): {e.stderr.strip()}",
                file=sys.stderr,
            )
            time.sleep(POLL_INTERVAL_S)
            continue

        gpus = parse_nvidia_smi(output)
        idx = pick_available(gpus)
        if idx is not None:
            print(idx)
            return

        for g in gpus:
            print(
                f"gpu {g.index}: util={g.util_pct}% free={g.free_mib} MiB — busy",
                file=sys.stderr,
            )
        print(f"no idle gpu; retrying in {POLL_INTERVAL_S}s", file=sys.stderr)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
