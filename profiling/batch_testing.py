#!/usr/bin/env python3
"""
Batch size latency test for Triton YOLO ONNX model.
Sends requests with batch sizes 1 to 8, 100 times each (sequential, no overlap),
and records the average response time per batch size. Results are written to a CSV file.

Usage:
  # Port-forward first, then from repo root:
  kubectl port-forward svc/triton-yolo 8000:8000
  uv run python profiling/batch_testing.py
  uv run python profiling/batch_testing.py --url http://localhost:8000 --output results.csv
"""
import argparse
import csv
import sys
import time

import numpy as np
import requests


def run_batch_requests(base_url: str, model: str, batch_size: int, num_requests: int) -> float:
    """
    Send num_requests inference requests with the given batch_size, one by one (no overlap).
    Returns average response time in milliseconds.
    """
    shape = [batch_size, 3, 640, 640]
    data = np.zeros(shape, dtype=np.float32)
    data_list = data.flatten().tolist()

    payload = {
        "inputs": [
            {
                "name": "images",
                "shape": shape,
                "datatype": "FP32",
                "data": data_list,
            }
        ],
        "outputs": [{"name": "output0"}],
    }

    infer_url = f"{base_url}/v2/models/{model}/infer"
    times_ms = []

    for _ in range(num_requests):
        start = time.perf_counter()
        r = requests.post(infer_url, json=payload, timeout=60)
        r.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

    return sum(times_ms) / len(times_ms)


def main():
    parser = argparse.ArgumentParser(description="Batch size latency test for Triton YOLO")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Triton HTTP endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="yolo_onnx",
        help="Model name (default: yolo_onnx)",
    )
    parser.add_argument(
        "--requests-per-batch",
        type=int,
        default=100,
        help="Number of requests to send per batch size (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="batch_test_results.csv",
        help="Output CSV file path (default: batch_test_results.csv)",
    )
    args = parser.parse_args()

    base = args.url.rstrip("/")

    # Health check
    try:
        r = requests.get(f"{base}/v2/health/ready", timeout=5)
        if r.status_code != 200:
            print(f"Server not ready: {r.status_code}", file=sys.stderr)
            sys.exit(1)
    except requests.RequestException as e:
        print(f"Cannot reach Triton at {base}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Testing batch sizes 1..8, {args.requests_per_batch} requests each (sequential).")
    print(f"Results will be written to: {args.output}\n")

    results = []

    for batch_size in range(1, 9):
        print(f"Batch size {batch_size}... ", end="", flush=True)
        try:
            avg_ms = run_batch_requests(
                base, args.model, batch_size, args.requests_per_batch
            )
            results.append((batch_size, avg_ms))
            print(f"avg response time: {avg_ms:.2f} ms")
        except requests.RequestException as e:
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "avg_response_time_ms"])
        writer.writerows(results)

    print(f"\nDone. Results saved to {args.output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
