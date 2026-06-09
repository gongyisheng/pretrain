"""Per-weight statistics for a single checkpoint.

Usage:
    python scripts/inspect_weights.py --ckpt <checkpoint.pt>
    python scripts/inspect_weights.py --ckpt checkpoints/grokking/qwen3_1m_sub_wd0.5_ce_fp64_lion_ls1e-5/30000.pt
    python scripts/inspect_weights.py --ckpt checkpoints/grokking/qwen3_1m_sub_wd0.5_ce_fp64_lion_ls1e-5/40000.pt
    python scripts/inspect_weights.py --ckpt checkpoints/grokking/qwen3_1m_sub_wd0.5_ce_fp64_lion_ls1e-5/50000.pt

Outputs max / min / mean / std / rms / p1 / p10 / p50 / p90 / p99 for every weight tensor, sortable by any stat.
"""
import argparse
import torch


PCTL_TENSOR = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99])
PCTL_KEYS = ["p1", "p10", "p50", "p90", "p99"]
STAT_KEYS = ["min"] + PCTL_KEYS + ["max", "mean", "std", "rms"]


def load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def tensor_stats(t: torch.Tensor) -> dict:
    f = t.float().flatten()
    nan_mask = torch.isnan(f)
    valid = f[~nan_mask]
    nan_count = nan_mask.sum().item()
    out = {"nans": nan_count, "numel": f.numel()}
    if valid.numel() == 0:
        for k in STAT_KEYS:
            out[k] = float("nan")
        return out
    out["max"] = valid.max().item()
    out["min"] = valid.min().item()
    out["mean"] = valid.mean().item()
    out["std"] = valid.std().item()
    out["rms"] = valid.pow(2).mean().sqrt().item()
    q = torch.quantile(valid, PCTL_TENSOR.to(valid.dtype))
    for k, v in zip(PCTL_KEYS, q.tolist()):
        out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Per-weight max/min/mean/std/percentiles for a checkpoint")
    parser.add_argument("--ckpt", required=True, help="Checkpoint file (.pt)")
    parser.add_argument("--sort", default="name", choices=STAT_KEYS + ["nans", "name"],
                        help="Sort by this stat (default: name)")
    parser.add_argument("--top",  type=int, default=0,
                        help="Show only top N rows (default: 0 = all)")
    parser.add_argument("--abs",  action="store_true",
                        help="Sort by absolute value of the stat")
    args = parser.parse_args()

    print(f"Loading {args.ckpt} ...")
    ckpt = load(args.ckpt)
    step = ckpt.get("step", "?")
    model_sd = ckpt.get("model", ckpt)  # support bare state_dict too

    rows = []
    for name, tensor in model_sd.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        s = tensor_stats(tensor)
        rows.append((name, s))

    if args.sort == "name":
        rows.sort(key=lambda r: r[0])
    else:
        key_fn = (lambda r: abs(r[1][args.sort])) if args.abs else (lambda r: r[1][args.sort])
        rows.sort(key=key_fn, reverse=(args.sort != "min"))

    display = rows[:args.top] if args.top > 0 else rows

    print(f"\nStep: {step}  |  {len(rows)} float tensors  |  sorted by {'|' + args.sort + '|' if args.abs else args.sort}\n")
    cols = STAT_KEYS
    hdr_stats = " ".join(f"{c:>9}" for c in cols)
    hdr = f"  {'parameter':<50} {hdr_stats}  {'nans':>5}  {'numel':>8}"
    print(hdr)
    print("  " + "-" * (50 + len(cols) * 10 + 18))
    for name, s in display:
        nan_flag = "  !" if s["nans"] > 0 else ""
        stat_vals = " ".join(f"{s[c]:9.5f}" for c in cols)
        print(f"  {name:<50} {stat_vals}  {s['nans']:>5}{nan_flag}  {s['numel']:>8,}")

    if args.top > 0 and args.top < len(rows):
        print(f"\n  ... ({len(rows) - args.top} more rows hidden, use --top 0 to show all)")

    print()


if __name__ == "__main__":
    main()
