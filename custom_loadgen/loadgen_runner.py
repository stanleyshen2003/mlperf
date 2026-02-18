"""
Priority-aware load generator: MLPerf LoadGen (Server scenario, Poisson arrivals)
driving traffic to the custom scheduler via gRPC. One LoadGen instance per
request class (tier, slo), run concurrently in separate processes (LoadGen C++
uses global singletons and is not safe to run multiple tests in parallel threads).
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

# Prefer repo loadgen when running from custom_loadgen (sibling directory)
_repo_loadgen = Path(__file__).resolve().parent.parent / "loadgen"
if _repo_loadgen.exists() and str(_repo_loadgen) not in sys.path:
    sys.path.insert(0, str(_repo_loadgen))

try:
    import mlperf_loadgen as lg
except ImportError:
    print(
        "Error: mlperf_loadgen not found. Build and install the loadgen from the repo.",
        file=sys.stderr,
    )
    print("  cd loadgen && pip install .   # or: uv pip install -e .", file=sys.stderr)
    sys.exit(1)

from workload_config import (
    REQUEST_CLASSES,
    SLO_LATENCY_NS,
    WORKLOAD_DISTRIBUTIONS,
)
from qsl import make_qsl
from sut import SchedulerSUT
from metrics import write_csv, print_summary


# Fixed default seed base; each (tier, slo) gets a distinct deterministic seed.
DEFAULT_SEED_BASE = 42
SLO_SEED_OFFSET = {"urgent": 0, "normal": 1, "relaxed": 2}


def _class_seed(tier: int, slo: str) -> int:
    """Fixed deterministic seed per traffic type (tier, slo)."""
    return DEFAULT_SEED_BASE + tier * 10 + SLO_SEED_OFFSET.get(slo, 0)


def run_class(
    tier: int,
    slo: str,
    class_qps: float,
    duration_ms: int,
    min_query_count: int,
    class_fraction: float,
    scheduler_url: str,
    output_dir: Path,
    result_queue: multiprocessing.Queue,
) -> None:
    """Run one LoadGen test for (tier, slo). Puts sut.records onto result_queue when done."""
    seed = _class_seed(tier, slo)
    sut = SchedulerSUT(scheduler_url, tier, slo, seed=seed)
    lg_sut = sut.build_sut()
    qsl = make_qsl()

    # Per-class minimum proportional to workload share so issued counts match spec ratio
    class_min = max(1, round(min_query_count * class_fraction))

    settings = lg.TestSettings()
    settings.scenario = lg.TestScenario.Server
    settings.mode = lg.TestMode.PerformanceOnly
    settings.server_target_qps = class_qps
    settings.server_target_latency_ns = SLO_LATENCY_NS[slo]
    settings.min_duration_ms = duration_ms
    settings.min_query_count = class_min

    log_output = lg.LogOutputSettings()
    log_output.outdir = str(output_dir / f"tier{tier}_{slo}")
    log_output.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output
    Path(log_output.outdir).mkdir(parents=True, exist_ok=True)

    lg.StartTestWithLogSettings(lg_sut, qsl, settings, log_settings)

    sut.destroy()
    lg.DestroyQSL(qsl)

    result_queue.put(sut.records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run priority-aware LoadGen (Server scenario) against scheduler gRPC."
    )
    parser.add_argument(
        "--scheduler-url",
        type=str,
        default="localhost:50051",
        help="Scheduler gRPC endpoint (default: localhost:50051)",
    )
    parser.add_argument(
        "--total-qps",
        type=float,
        default=30.0,
        help="Total aggregate QPS across all classes (default: 30)",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=60000,
        help="Test duration in ms (default: 60000)",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="balanced",
        choices=list(WORKLOAD_DISTRIBUTIONS),
        help="Workload distribution: balanced, urgent_heavy, batch_heavy (default: balanced)",
    )
    parser.add_argument(
        "--min-query-count",
        type=int,
        default=100,
        help="Min query count (distributed per class by workload fraction so ratio matches spec) (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result/loadgen_logs",
        help="Root dir for LoadGen internal logs (default: result/loadgen_logs)",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="result/metrics.csv",
        help="Per-request output CSV (default: result/metrics.csv)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dist = WORKLOAD_DISTRIBUTIONS[args.workload]
    result_queue = multiprocessing.Queue()
    processes = []

    for tier, slo in REQUEST_CLASSES:
        class_fraction = dist.get((tier, slo), 0.0)
        class_qps = args.total_qps * class_fraction
        if class_qps < 0.01:
            continue
        p = multiprocessing.Process(
            target=run_class,
            args=(
                tier,
                slo,
                class_qps,
                args.duration_ms,
                args.min_query_count,
                class_fraction,
                args.scheduler_url,
                output_dir,
                result_queue,
            ),
        )
        processes.append(p)
        p.start()

    all_records = []
    for _ in processes:
        all_records.extend(result_queue.get())

    for p in processes:
        p.join()

    write_csv(all_records, args.metrics_csv)
    print_summary(all_records)

    if args.metrics_csv:
        print(f"\nMetrics CSV: {Path(args.metrics_csv).resolve()} ({len(all_records)} requests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
