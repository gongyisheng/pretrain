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

sys.path.insert(0, ".")

from src.utils.gpu_select import parse_nvidia_smi, pick_available

POLL_INTERVAL_S = 30


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
