"""
Per-request logging and CSV output for loadgen runs.
"""

import csv
from collections import defaultdict

from workload_config import SLO_LATENCY_NS


def write_csv(records: list, output_path: str) -> None:
    """
    Write records to CSV with columns:
    timestamp_issued, sample_id, tier, slo, model, latency_ns, latency_ms, slo_met
    """
    if not records:
        return
    fieldnames = [
        "timestamp_issued",
        "sample_id",
        "tier",
        "slo",
        "model",
        "latency_ns",
        "latency_ms",
        "slo_met",
    ]
    # slo_met is already in record; ensure we output as model (from model_name)
    rows = []
    for r in records:
        rows.append({
            "timestamp_issued": r["timestamp_issued"],
            "sample_id": r["sample_id"],
            "tier": r["tier"],
            "slo": r["slo"],
            "model": r["model_name"],
            "latency_ns": r["latency_ns"],
            "latency_ms": r["latency_ns"] / 1e6,
            "slo_met": r["slo_met"],
        })
    path = __import__("pathlib").Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_summary(records: list) -> None:
    """
    Print a summary table grouped by (tier, slo) with:
    count, p50_ms, p95_ms, p99_ms, slo_attainment_%
    """
    if not records:
        print("No records to summarize.")
        return

    # Group by (tier, slo)
    groups = defaultdict(list)
    for r in records:
        key = (r["tier"], r["slo"])
        groups[key].append(r["latency_ns"])

    # Sort keys for stable output
    keys = sorted(groups.keys())
    slo_ns = SLO_LATENCY_NS

    print("\nSummary by (tier, slo):")
    print(f"{'tier':<6} {'slo':<10} {'count':<8} {'p50_ms':<10} {'p95_ms':<10} {'p99_ms':<10} {'slo_attainment_%':<18}")
    print("-" * 72)

    for (tier, slo) in keys:
        latencies_ns = groups[(tier, slo)]
        n = len(latencies_ns)
        target_ns = slo_ns[slo]
        met = sum(1 for x in latencies_ns if x <= target_ns)
        attainment = 100.0 * met / n if n else 0.0

        sorted_ns = sorted(latencies_ns)
        idx50 = min(int(0.50 * n), n - 1) if n else 0
        idx95 = min(int(0.95 * n), n - 1) if n else 0
        idx99 = min(int(0.99 * n), n - 1) if n else 0
        p50 = sorted_ns[idx50] / 1e6 if n else 0.0
        p95 = sorted_ns[idx95] / 1e6 if n else 0.0
        p99 = sorted_ns[idx99] / 1e6 if n else 0.0

        print(f"{tier:<6} {slo:<10} {n:<8} {p50:<10.2f} {p95:<10.2f} {p99:<10.2f} {attainment:<18.1f}")

    print("-" * 72)
