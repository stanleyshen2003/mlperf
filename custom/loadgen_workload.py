"""
Mixed ML inference workload using MLPerf LoadGen.

Sends a configurable mix of ResNet, YOLO, and BERT queries to the fake
inference server. Supports fixed target QPS or changeable QPS via --schedule
(e.g. 0-5s @ 3 QPS, 5-10s @ 10 QPS, 10-15s @ 20 QPS) by running LoadGen
once per segment.

Use --kserve to send traffic to a KServe v2 infer endpoint (e.g. MNIST)
instead of the fake server. With --kserve, requests are sent to
/v2/models/<model>/infer with a JSON payload (default: infer_v2.json in
repo root).

Requires: loadgen built and installed from the repo (see README).
  cd loadgen && pip install .   # or: uv pip install -e . from repo root
  Then run from repo root: uv run python custom/loadgen_workload.py [options]
"""

import argparse
import csv
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Prefer repo loadgen when running from custom/ (sibling of loadgen at repo root)
_repo_loadgen = Path(__file__).resolve().parent.parent / "loadgen"
if _repo_loadgen.exists() and str(_repo_loadgen) not in sys.path:
    sys.path.insert(0, str(_repo_loadgen))

try:
    import mlperf_loadgen as lg
except ImportError:
    print(
        "Error: mlperf_loadgen not found. Build and install the loadgen from the repo:",
        file=sys.stderr,
    )
    print("  cd loadgen && pip install .   # or: uv pip install -e .", file=sys.stderr)
    print("  Then use the same Python/venv to run this script.", file=sys.stderr)
    sys.exit(1)

import requests

# Fake server endpoints (used when not --kserve)
ENDPOINTS = ("resnet", "yolo", "bert")


def parse_schedule(schedule_str: str) -> List[Tuple[float, float, float]]:
    """Parse --schedule '0-5:3,5-10:10,10-15:20' into [(t_start, t_end, qps), ...]."""
    segments = []
    for part in schedule_str.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"Invalid segment '{part}': expected 'start-end:qps'")
        range_str, qps_str = part.rsplit(":", 1)
        if "-" not in range_str:
            raise ValueError(f"Invalid range '{range_str}': expected 'start-end'")
        t_start_str, t_end_str = range_str.strip().split("-", 1)
        t_start = float(t_start_str.strip())
        t_end = float(t_end_str.strip())
        qps = float(qps_str.strip())
        if t_start >= t_end or qps < 0:
            raise ValueError("Invalid segment: start < end and qps >= 0 required")
        segments.append((t_start, t_end, qps))
    segments.sort(key=lambda x: x[0])
    return segments


def _endpoint_for_index(index: int, n_resnet: int, n_yolo: int, n_bert: int) -> str:
    """Map QSL sample index to endpoint name.
    Uses interleaved mapping (index % 3) so that when LoadGen only uses
    the first performance_sample_count indices (e.g. 0-499), we still get
    a mix of resnet/yolo/bert. Range-based mapping would put all 0-499 in
    resnet and yield 100% resnet requests.
    """
    # Interleave by index so any contiguous block of indices gets even mix
    which = index % 3
    if which == 0:
        return "resnet"
    if which == 1:
        return "yolo"
    return "bert"


def _send_one(base_url: str, endpoint: str, timeout: float, tag: Optional[int] = None) -> bool:
    url = f"{base_url.rstrip('/')}/{endpoint}/"
    payload = {} if tag is None else {"tag": tag}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _send_one_kserve(
    base_url: str,
    model_name: str,
    payload: dict,
    timeout: float,
    host_header: Optional[str] = None,
) -> bool:
    """Send one inference request to KServe v2 infer endpoint."""
    url = f"{base_url.rstrip('/')}/v2/models/{model_name}/infer"
    headers = {"Content-Type": "application/json"}
    if host_header:
        headers["Host"] = host_header
    try:
        r = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


def make_issue_query(
    base_url: str,
    n_resnet: int,
    n_yolo: int,
    n_bert: int,
    timeout: float,
    workload_records: Optional[list] = None,
    workload_lock: Optional[threading.Lock] = None,
):
    """Build issue_query callback that dispatches to fake server by sample index.
    If workload_records and workload_lock are provided, each request is logged
    with timestamp_issued, response_id, sample_index, model, latency_ns.
    """

    def issue_query(query_samples):
        # Copy list so loadgen can reuse its buffers
        samples = [(s.id, s.index) for s in query_samples]

        def run():
            responses = []
            for sample_id, index in samples:
                tag = random.randint(1, 3)
                endpoint = _endpoint_for_index(index, n_resnet, n_yolo, n_bert)
                t_issued = time.time()
                t0 = time.perf_counter()
                _send_one(base_url, endpoint, timeout, tag=tag)
                t1 = time.perf_counter()
                latency_ns = int((t1 - t0) * 1e9)
                if workload_records is not None and workload_lock is not None:
                    with workload_lock:
                        workload_records.append({
                            "timestamp_issued": t_issued,
                            "response_id": sample_id,
                            "sample_index": index,
                            "model": endpoint,
                            "tag": tag,
                            "kserve": False,
                            "latency_ns": latency_ns,
                            "latency_ms": (t1 - t0) * 1000,
                        })
                responses.append(lg.QuerySampleResponse(sample_id, 0, 0))
            lg.QuerySamplesComplete(responses)

        threading.Thread(target=run).start()

    return issue_query


def make_issue_query_kserve(
    base_url: str,
    model_name: str,
    payload: dict,
    timeout: float,
    host_header: Optional[str] = None,
    workload_records: Optional[list] = None,
    workload_lock: Optional[threading.Lock] = None,
):
    """Build issue_query callback that sends KServe v2 infer requests (single model, same payload)."""

    def issue_query(query_samples):
        samples = [(s.id, s.index) for s in query_samples]

        def run():
            responses = []
            for sample_id, index in samples:
                t_issued = time.time()
                t0 = time.perf_counter()
                ok = _send_one_kserve(base_url, model_name, payload, timeout, host_header=host_header)
                t1 = time.perf_counter()
                latency_ns = int((t1 - t0) * 1e9)
                if workload_records is not None and workload_lock is not None:
                    with workload_lock:
                        workload_records.append({
                            "timestamp_issued": t_issued,
                            "response_id": sample_id,
                            "sample_index": index,
                            "model": model_name,
                            "kserve": True,
                            "latency_ns": latency_ns,
                            "latency_ms": (t1 - t0) * 1000,
                        })
                responses.append(lg.QuerySampleResponse(sample_id, 0, 0))
            lg.QuerySamplesComplete(responses)

        threading.Thread(target=run).start()

    return issue_query


def main():
    parser = argparse.ArgumentParser(
        description="Run MLPerf LoadGen mixed workload (ResNet + YOLO + BERT) against fake server."
    )
    parser.add_argument(
        "--kserve",
        action="store_true",
        help="Send traffic to KServe v2 infer API (default URL http://127.0.0.1:8080, payload from infer_v2.json)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of server (default for --kserve: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--kserve-model",
        type=str,
        default="mnist",
        help="KServe model name for /v2/models/<name>/infer (default: mnist). Used only when --kserve.",
    )
    parser.add_argument(
        "--infer-payload",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to JSON payload for KServe infer (default: infer_v2.json in repo root). Used only when --kserve.",
    )
    parser.add_argument(
        "--kserve-host-header",
        type=str,
        default="mlflow-v2-wine-classifier-predictor.default.example.com",
        help="Host header for KServe v2 infer requests (default: mlflow-v2-wine-classifier-predictor.default.example.com). Used only when --kserve.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["Server", "Offline"],
        default="Server",
        help="LoadGen scenario (default: Server)",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=10000,
        help="Min duration in ms (default: 10000)",
    )
    parser.add_argument(
        "--target-qps",
        type=float,
        default=10.0,
        help="Target QPS for Server scenario (default: 10)",
    )
    parser.add_argument(
        "--min-query-count",
        type=int,
        default=100,
        help="Min query count (default: 100)",
    )
    parser.add_argument(
        "--mix",
        type=str,
        default="1,1,1",
        help="Sample count ratio resnet,yolo,bert (default: 1,1,1). Total samples = sum; partition by this ratio.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=3000,
        help="Total QSL sample count (default: 3000). Partitioned by --mix.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for LoadGen logs (default: loadgen_logs under script dir)",
    )
    parser.add_argument(
        "--workload-log",
        type=str,
        default=None,
        metavar="PATH",
        help="Write per-request workload CSV: timestamp_issued, response_id, sample_index, model, tag (1-3), latency_ns, latency_ms",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        metavar="SPEC",
        help="Time-varying QPS: run LoadGen per segment. Format 'start-end:qps,...' e.g. '0-5:3,5-10:10,10-15:20' (0-5s @ 3 QPS, 5-10s @ 10 QPS, 10-15s @ 20 QPS). Overrides --duration-ms and --target-qps.",
    )
    args = parser.parse_args()

    # When --kserve, default URL to KServe port if user did not change it
    if args.kserve and args.url == "http://127.0.0.1:8000":
        args.url = "http://127.0.0.1:8080"

    if args.output_dir is None:
        # Use temp dir with ASCII path to avoid loadgen C++ issues on Windows Unicode paths
        import tempfile
        args.output_dir = Path(tempfile.gettempdir()) / "fakeserver_loadgen_logs"
    else:
        args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(args.output_dir)

    if args.kserve:
        # KServe mode: single model, payload from JSON file
        infer_payload_path = Path(args.infer_payload) if args.infer_payload else (Path(__file__).resolve().parent.parent / "infer_v2.json")
        if not infer_payload_path.exists():
            print(f"Error: KServe infer payload not found at {infer_payload_path}", file=sys.stderr)
            sys.exit(1)
        with open(infer_payload_path, "r") as f:
            kserve_payload = json.load(f)
        n_resnet = 0
        n_yolo = 0
        n_bert = 0
        total = args.total_samples
    else:
        kserve_payload = None
        parts = [float(x.strip()) for x in args.mix.split(",")]
        if len(parts) != 3 or any(p < 0 for p in parts) or sum(parts) <= 0:
            print("Error: --mix must be three non-negative numbers, e.g. 1,1,1", file=sys.stderr)
            sys.exit(1)
        total_weight = sum(parts)
        n_resnet = int(round(args.total_samples * parts[0] / total_weight))
        n_yolo = int(round(args.total_samples * parts[1] / total_weight))
        n_bert = args.total_samples - n_resnet - n_yolo
        if n_bert < 0:
            n_bert = 0
        total = n_resnet + n_yolo + n_bert

    # Health check
    if args.kserve:
        try:
            r = requests.get(f"{args.url.rstrip('/')}/v2/health/ready", timeout=5)
            if r.status_code != 200:
                print(f"Warning: KServe /v2/health/ready returned {r.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: could not reach KServe at {args.url} (v2/health/ready): {e}", file=sys.stderr)
            print("  Proceeding anyway; first infer request may fail.", file=sys.stderr)
    else:
        try:
            r = requests.get(f"{args.url.rstrip('/')}/health/", timeout=5)
            if r.status_code != 200:
                print(f"Warning: /health/ returned {r.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"Error: cannot reach server at {args.url}: {e}", file=sys.stderr)
            sys.exit(1)

    # Optional: time-varying QPS schedule
    if args.schedule:
        try:
            segments = parse_schedule(args.schedule)
        except ValueError as e:
            print(f"Error: invalid --schedule: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        segments = None

    # QSL: no actual data to load; just satisfy LoadGen's count contract
    def load_samples_to_ram(query_sample_indices):
        pass

    def unload_samples_from_ram(query_sample_indices):
        pass

    perf_count = min(500, total)
    qsl = lg.ConstructQSL(total, perf_count, load_samples_to_ram, unload_samples_from_ram)

    # Optional per-request workload log (timestamp, model, latency per request)
    workload_records = []
    workload_lock = threading.Lock()
    if args.kserve:
        issue_query = make_issue_query_kserve(
            args.url,
            args.kserve_model,
            kserve_payload,
            args.timeout,
            host_header=args.kserve_host_header,
            workload_records=workload_records if (args.workload_log or segments) else None,
            workload_lock=workload_lock if (args.workload_log or segments) else None,
        )
    else:
        issue_query = make_issue_query(
            args.url,
            n_resnet,
            n_yolo,
            n_bert,
            args.timeout,
            workload_records=workload_records if (args.workload_log or segments) else None,
            workload_lock=workload_lock if (args.workload_log or segments) else None,
        )

    def flush_queries():
        pass

    sut = lg.ConstructSUT(issue_query, flush_queries)

    # Logging
    log_output = lg.LogOutputSettings()
    log_output.outdir = args.output_dir
    log_output.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output

    if segments:
        # Run LoadGen once per schedule segment (changeable QPS)
        total_duration_sec = segments[-1][1]
        title = "LoadGen KServe workload (scheduled)" if args.kserve else "LoadGen mixed workload (scheduled)"
        print(f"{title}: {args.scenario}")
        print(f"  Server: {args.url}")
        if args.kserve:
            print(f"  KServe model: {args.kserve_model}, samples: {total}")
        else:
            print(f"  Mix (samples): resnet={n_resnet}, yolo={n_yolo}, bert={n_bert} (total={total})")
        print(f"  Schedule: {total_duration_sec:.1f}s total")
        for t_start, t_end, qps in segments:
            print(f"    {t_start:.1f}s - {t_end:.1f}s  QPS = {qps}")
        print(f"  Logs: {args.output_dir}")
        print()
        for i, (t_start, t_end, qps) in enumerate(segments):
            duration_ms = int((t_end - t_start) * 1000)
            settings = lg.TestSettings()
            settings.mode = lg.TestMode.PerformanceOnly
            settings.scenario = lg.TestScenario.Server
            settings.server_target_qps = qps
            settings.server_target_latency_ns = 500_000_000
            settings.min_duration_ms = duration_ms
            settings.min_query_count = max(1, int(qps * (t_end - t_start) * 0.5))
            print(f"  Phase {i + 1}/{len(segments)}: {t_start:.1f}s - {t_end:.1f}s @ {qps} QPS ({duration_ms} ms)")
            lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
        lg.DestroySUT(sut)
        lg.DestroyQSL(qsl)
        workload_log_path = args.workload_log or "workload_schedule.csv"
    else:
        # Single phase: fixed duration and target QPS
        settings = lg.TestSettings()
        settings.mode = lg.TestMode.PerformanceOnly
        settings.min_duration_ms = args.duration_ms
        settings.min_query_count = args.min_query_count
        if args.scenario == "Server":
            settings.scenario = lg.TestScenario.Server
            settings.server_target_qps = args.target_qps
            settings.server_target_latency_ns = 500_000_000  # 500 ms
        else:
            settings.scenario = lg.TestScenario.Offline
            settings.offline_expected_qps = args.target_qps
        title = "LoadGen KServe workload" if args.kserve else "LoadGen mixed workload"
        print(f"{title}: {args.scenario}")
        print(f"  Server: {args.url}")
        if args.kserve:
            print(f"  KServe model: {args.kserve_model}, samples: {total}")
        else:
            print(f"  Mix (samples): resnet={n_resnet}, yolo={n_yolo}, bert={n_bert} (total={total})")
        print(f"  Target QPS: {args.target_qps}, duration >= {args.duration_ms} ms")
        print(f"  Logs: {args.output_dir}")
        print()
        try:
            lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
        finally:
            lg.DestroySUT(sut)
            lg.DestroyQSL(qsl)
        workload_log_path = args.workload_log

    # Write workload log so you can see actual model inference workload (timestamp, model, latency)
    if workload_log_path and workload_records:
        out_path = Path(workload_log_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp_issued",
                    "response_id",
                    "sample_index",
                    "model",
                    "tag",
                    "kserve",
                    "latency_ns",
                    "latency_ms",
                ],
            )
            w.writeheader()
            w.writerows(workload_records)
        print(f"Workload log written to {out_path} ({len(workload_records)} requests)")


if __name__ == "__main__":
    main()
