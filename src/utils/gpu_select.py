"""Pure helpers for selecting an idle GPU.

`parse_nvidia_smi` parses the CSV output of
    nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
               --format=csv,noheader,nounits
into typed records. `pick_available` selects the lowest-index GPU whose
utilization and free memory both clear the thresholds.

Kept side-effect-free so it can be unit-tested without calling nvidia-smi.
The CLI wrapper that actually invokes nvidia-smi lives in
scripts/pick_free_gpu.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    util_pct: int
    free_mib: int


def parse_nvidia_smi(output: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        idx, util, free = (part.strip() for part in line.split(","))
        gpus.append(GpuInfo(index=int(idx), util_pct=int(util), free_mib=int(free)))
    return gpus


def pick_available(
    gpus: list[GpuInfo],
    max_util_pct: int = 10,
    min_free_mib: int = 8 * 1024,
) -> int | None:
    for g in gpus:
        if g.util_pct <= max_util_pct and g.free_mib >= min_free_mib:
            return g.index
    return None
