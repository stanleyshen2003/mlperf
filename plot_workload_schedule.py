"""
Plot workload_schedule.csv by timestamp_issued: QPS over time and request timeline.
Run from this directory: python plot_workload_schedule.py [workload_schedule.csv]
Requires: pip install matplotlib
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Plot workload_schedule.csv by timestamp_issued")
    parser.add_argument(
        "csv",
        nargs="?",
        default="workload_schedule.csv",
        help="Path to workload CSV (default: workload_schedule.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output image path (default: workload_schedule.png)",
    )
    parser.add_argument(
        "--bin-sec",
        type=float,
        default=1.0,
        help="Time bin size in seconds for QPS plot (default: 1.0)",
    )
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows or "timestamp_issued" not in rows[0]:
        raise SystemExit("CSV must contain column 'timestamp_issued'")

    t = [float(r["timestamp_issued"]) for r in rows]
    t0 = min(t)
    t_rel = [x - t0 for x in t]
    models = [r["model"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 1) QPS over time (binned by timestamp_issued)
    t_max = max(t_rel)
    bins = []
    x = 0.0
    while x <= t_max + args.bin_sec:
        bins.append(x)
        x += args.bin_sec
    counts = [0] * (len(bins) - 1)
    for tr in t_rel:
        i = int(tr / args.bin_sec)
        if 0 <= i < len(counts):
            counts[i] += 1
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    qps = [c / args.bin_sec for c in counts]
    axes[0].bar(bin_centers, qps, width=args.bin_sec * 0.85, align="center", color="steelblue", edgecolor="white")
    axes[0].set_ylabel("QPS")
    axes[0].set_title("Request rate (QPS) over time (from first request)")
    axes[0].grid(True, alpha=0.3)

    # 2) Request timeline: one point per request, colored by model
    colors = {"resnet": "C0", "yolo": "C1", "bert": "C2"}
    for model in ("resnet", "yolo", "bert"):
        t_m = [t_rel[i] for i in range(len(t_rel)) if models[i] == model]
        if t_m:
            axes[1].scatter(t_m, [1] * len(t_m), c=colors[model], label=model, alpha=0.6, s=12)
    axes[1].set_xlabel("Time (s from first request)")
    axes[1].set_ylabel("Request")
    axes[1].set_title("Requests over time (by model)")
    axes[1].set_yticks([])
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.output or "workload_schedule.png").resolve()
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
