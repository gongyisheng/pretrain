#!/usr/bin/env python3
"""Reserve an idle GPU, then run a command pinned to it.

    gpu_lock_exec.py -- pytest tests/fast/layers

Polls until a GPU is free, then runs the command with `CUDA_VISIBLE_DEVICES`
set to it and exits with the child's status. "Free" means two things, because
they catch different squatters: an flock excludes other CI jobs, including in
the seconds before their memory reaches `nvidia-smi`, and the nvidia-smi
thresholds exclude training runs started outside CI.

Waits indefinitely; the workflow's `timeout-minutes` bounds the wait.
"""

import fcntl
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

POLL_INTERVAL_S = 30
LOCK_DIR = Path(os.environ.get("GPU_LOCK_DIR", "/tmp/pretrain-gpu-locks"))


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


def reserve(
    gpus: list[GpuInfo],
    max_util_pct: int = 10,
    min_free_mib: int = 8 * 1024,  # 8 GiB
) -> tuple[int, int] | None:
    """Lock and return the lowest-index idle GPU as (index, held fd), else None.

    The fd comes back open because closing it drops the lock.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    for gpu in gpus:
        fd = os.open(LOCK_DIR / f"{gpu.index}.lock", os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            print(f"gpu {gpu.index}: locked by another job", file=sys.stderr)
            continue
        if gpu.util_pct <= max_util_pct and gpu.free_mib >= min_free_mib:
            return gpu.index, fd
        # Idle enough to lock, busy enough to skip: something outside CI has it.
        os.close(fd)
        print(
            f"gpu {gpu.index}: util={gpu.util_pct}% free={gpu.free_mib} MiB — busy",
            file=sys.stderr,
        )
    return None


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(f"usage: {sys.argv[0]} -- COMMAND [ARGS...]", file=sys.stderr)
        return 2

    while True:
        try:
            gpus = parse_nvidia_smi(query_nvidia_smi())
        except subprocess.CalledProcessError as e:
            print(
                f"nvidia-smi failed (rc={e.returncode}): {e.stderr.strip()}",
                file=sys.stderr,
            )
            time.sleep(POLL_INTERVAL_S)
            continue

        reserved = reserve(gpus)
        if reserved is not None:
            break
        print(f"no idle gpu; retrying in {POLL_INTERVAL_S}s", file=sys.stderr)
        time.sleep(POLL_INTERVAL_S)

    index, _fd = reserved  # _fd stays open so the lock outlives the poll loop
    print(f"reserved gpu {index} for: {' '.join(argv)}", file=sys.stderr)
    return subprocess.run(
        argv, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(index)}
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
