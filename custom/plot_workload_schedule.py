"""
Plot workload_schedule.csv by timestamp_issued: QPS over time, average latency over time,
per-request latency over time, and request timeline.
Run from custom/: python plot_workload_schedule.py [workload_schedule.csv]
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
        default=0.5,
        help="Time bin size in seconds for QPS and average latency (default: 0.5)",
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
    if "latency_ms" not in rows[0]:
        raise SystemExit("CSV must contain column 'latency_ms' for latency plots")

    t = [float(r["timestamp_issued"]) for r in rows]
    t0 = min(t)
    t_rel = [x - t0 for x in t]
    models = [r["model"] for r in rows]
    latency_ms = [float(r["latency_ms"]) for r in rows]

    t_max = max(t_rel)
    bins = []
    x = 0.0
    while x <= t_max + args.bin_sec:
        bins.append(x)
        x += args.bin_sec
    n_bins = len(bins) - 1
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # 1) QPS over time (binned by timestamp_issued)
    counts = [0] * n_bins
    for tr in t_rel:
        i = int(tr / args.bin_sec)
        if 0 <= i < n_bins:
            counts[i] += 1
    qps = [c / args.bin_sec for c in counts]
    axes[0].bar(bin_centers, qps, width=args.bin_sec * 0.85, align="center", color="steelblue", edgecolor="white")
    axes[0].set_ylabel("QPS")
    axes[0].set_title("Request rate (QPS) over time (from first request)")
    axes[0].grid(True, alpha=0.3)

    # 2) Average latency over time (binned)
    bin_latencies = [[] for _ in range(n_bins)]
    for tr, lat in zip(t_rel, latency_ms):
        i = int(tr / args.bin_sec)
        if 0 <= i < n_bins:
            bin_latencies[i].append(lat)
    avg_latency = [sum(l) / len(l) if l else float("nan") for l in bin_latencies]
    axes[1].plot(bin_centers, avg_latency, color="darkgreen", marker="o", markersize=3, linewidth=1.2)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Average latency over time (per-bin mean)")
    axes[1].grid(True, alpha=0.3)

    # 3) Latency of each request over time (one point per request)
    unique_models = list(dict.fromkeys(models))
    color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, model in enumerate(unique_models):
        t_m = [t_rel[i] for i in range(len(t_rel)) if models[i] == model]
        lat_m = [latency_ms[i] for i in range(len(t_rel)) if models[i] == model]
        if t_m:
            c = color_list[idx % len(color_list)]
            axes[2].scatter(t_m, lat_m, c=c, label=model, alpha=0.6, s=8)
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_title("Latency of each request over time")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # 4) Request timeline: one point per request, colored by model
    for idx, model in enumerate(unique_models):
        t_m = [t_rel[i] for i in range(len(t_rel)) if models[i] == model]
        if t_m:
            c = color_list[idx % len(color_list)]
            axes[3].scatter(t_m, [1] * len(t_m), c=c, label=model, alpha=0.6, s=12)
    axes[3].set_xlabel("Time (s from first request)")
    axes[3].set_ylabel("Request")
    axes[3].set_title("Requests over time (by model)")
    axes[3].set_yticks([])
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.output or "workload_schedule.png").resolve()
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
