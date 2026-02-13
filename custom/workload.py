"""
Mixed ML inference workload: sends a configurable mix of requests to
ResNet, YOLO, and BERT endpoints on the fake inference server.
"""

import argparse
import random
import statistics
import sys
import threading
import time
from collections import defaultdict

import requests

# Endpoints on the fake server
ENDPOINTS = ("resnet", "yolo", "bert")


def send_request(base_url: str, endpoint: str, timeout: float) -> tuple:
    """Send one request to an endpoint. Returns (endpoint, latency_sec, success)."""
    url = f"{base_url.rstrip('/')}/{endpoint}/"
    start = time.perf_counter()
    try:
        r = requests.post(url, json={}, timeout=timeout)
        ok = r.status_code == 200
    except Exception:
        ok = False
    latency = time.perf_counter() - start
    return endpoint, latency, ok


def worker(
    base_url: str,
    timeout: float,
    mix_weights: tuple,
    stop_event: threading.Event,
    latencies: dict,
    counts: dict,
    errors: dict,
    lock: threading.Lock,
):
    """Worker thread: repeatedly pick an endpoint by mix and send a request."""
    while not stop_event.is_set():
        endpoint = random.choices(ENDPOINTS, weights=mix_weights, k=1)[0]
        ep, lat, ok = send_request(base_url, endpoint, timeout)
        with lock:
            latencies[ep].append(lat)
            counts[ep] += 1
            if not ok:
                errors[ep] = errors.get(ep, 0) + 1


def percentile(data: list, p: float) -> float:
    """Compute p-th percentile (0..100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_workload(
    base_url: str,
    duration_sec: float,
    concurrency: int,
    mix: tuple,
    timeout: float,
) -> None:
    """Run mixed workload for given duration with N concurrent workers."""
    latencies = defaultdict(list)
    counts = defaultdict(int)
    errors = defaultdict(int)
    lock = threading.Lock()
    stop_event = threading.Event()

    threads = [
        threading.Thread(
            target=worker,
            args=(base_url, timeout, mix, stop_event, latencies, counts, errors, lock),
        )
        for _ in range(concurrency)
    ]
    for t in threads:
        t.start()

    time.sleep(duration_sec)
    stop_event.set()
    for t in threads:
        t.join()

    # Summary
    total = sum(counts.values())
    total_errors = sum(errors.values())
    elapsed = duration_sec
    qps = total / elapsed if elapsed > 0 else 0

    print("=" * 60)
    print("Mixed workload summary")
    print("=" * 60)
    print(f"Server:     {base_url}")
    print(f"Duration:   {elapsed:.1f}s  |  Concurrency: {concurrency}")
    print(f"Mix (R:Y:B): {mix[0]:.2f} : {mix[1]:.2f} : {mix[2]:.2f}")
    print(f"Total:      {total} requests  |  {qps:.1f} QPS  |  Errors: {total_errors}")
    print()

    for ep in ENDPOINTS:
        L = latencies[ep]
        c = counts[ep]
        e = errors.get(ep, 0)
        if not L:
            print(f"  {ep:8}  (no requests)")
            continue
        p50 = percentile(L, 50)
        p95 = percentile(L, 95)
        p99 = percentile(L, 99)
        avg = statistics.mean(L)
        ep_qps = c / elapsed if elapsed > 0 else 0
        print(f"  {ep:8}  count={c:6}  QPS={ep_qps:6.1f}  err={e}")
        print(f"           lat ms: avg={avg*1000:6.1f}  p50={p50*1000:6.1f}  p95={p95*1000:6.1f}  p99={p99*1000:6.1f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run mixed ResNet/YOLO/BERT workload against fake inference server."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the fake server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Run workload for this many seconds (default: 10)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Number of concurrent worker threads (default: 6)",
    )
    parser.add_argument(
        "--mix",
        type=str,
        default="1,1,1",
        help="Request mix resnet,yolo,bert as relative weights (default: 1,1,1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    args = parser.parse_args()

    parts = [float(x.strip()) for x in args.mix.split(",")]
    if len(parts) != 3 or any(p < 0 for p in parts) or sum(parts) <= 0:
        print("Error: --mix must be three non-negative numbers, e.g. 1,1,1 or 0.5,0.3,0.2", file=sys.stderr)
        sys.exit(1)
    mix = tuple(parts)

    # Quick connectivity check
    try:
        r = requests.get(f"{args.url.rstrip('/')}/health/", timeout=5)
        if r.status_code != 200:
            print(f"Warning: /health/ returned {r.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"Error: cannot reach server at {args.url}: {e}", file=sys.stderr)
        print("Start the fake server first (e.g. make start or uv run python run.py).", file=sys.stderr)
        sys.exit(1)

    run_workload(
        base_url=args.url,
        duration_sec=args.duration,
        concurrency=args.concurrency,
        mix=mix,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
