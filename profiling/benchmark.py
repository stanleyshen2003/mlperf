#!/usr/bin/env python3
"""
Throughput benchmark for Triton YOLO ONNX model (saturation / Option B).
For each batch size (1, 2, 4, 8), runs for a fixed duration. Caps in-flight
images so each batch size is tested under the same load (fair comparison).
Only counts images from requests that completed before the deadline.

Usage:
  # Port-forward first, then from repo root:
  kubectl port-forward svc/triton-yolo 8000:8000
  uv run python profiling/benchmark.py
  uv run python profiling/benchmark.py --duration 60 --max-inflight 64
"""
import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
except ImportError:
    print("Install numpy: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)


# Batch sizes to test (1 -> 8 then 8 -> 1)
BATCH_SIZES = [1, 2, 4, 8]


def do_one_request(infer_url: str, payload: dict, timeout: int = 120) -> bool:
    """Send one inference request. Returns True on success."""
    try:
        r = requests.post(infer_url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True
    except requests.RequestException:
        return False


def run_saturation_test(
    base_url: str,
    model: str,
    batch_size: int,
    duration_sec: float,
    num_workers: int,
    max_inflight_images: int,
) -> tuple[int, float, float]:
    """
    Run for duration_sec with num_workers threads. Limits in-flight images to
    max_inflight_images so each batch size is under the same load (fair comparison).
    Only counts images from requests that completed before the deadline.
    Returns (images_in_window, window_sec, throughput_images_per_sec).
    """
    shape = [batch_size, 3, 640, 640]
    data = np.zeros(shape, dtype=np.float32)
    payload = {
        "inputs": [
            {
                "name": "images",
                "shape": shape,
                "datatype": "FP32",
                "data": data.flatten().tolist(),
            }
        ],
        "outputs": [{"name": "output0"}],
    }
    infer_url = f"{base_url}/v2/models/{model}/infer"

    completions: list[tuple[float, int]] = []  # (finish_time, batch_size)
    lock = threading.Lock()
    deadline = [0.0]  # mutable so worker can read
    # Cap in-flight images so we compare batch sizes under same load
    sem = threading.Semaphore(max_inflight_images)

    def worker() -> None:
        while time.perf_counter() < deadline[0]:
            # Reserve batch_size "slots" (blocks until available)
            for _ in range(batch_size):
                sem.acquire()
            try:
                if do_one_request(infer_url, payload):
                    finish = time.perf_counter()
                    with lock:
                        completions.append((finish, batch_size))
            finally:
                for _ in range(batch_size):
                    sem.release()

    start = time.perf_counter()
    deadline[0] = start + duration_sec
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker) for _ in range(num_workers)]
        for f in as_completed(futures):
            f.result()

    # Count only images from requests that finished before the deadline
    cutoff = start + duration_sec
    images_in_window = sum(bs for (ts, bs) in completions if ts <= cutoff)
    throughput = images_in_window / duration_sec if duration_sec > 0 else 0.0
    return images_in_window, duration_sec, throughput


def main():
    parser = argparse.ArgumentParser(description="Triton YOLO throughput benchmark (saturation)")
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
        "--duration",
        type=float,
        default=30.0,
        help="Seconds to run each batch size (default: 30)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Concurrent worker threads (default: 64)",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=64,
        help="Max images in flight at once; keeps load equal across batch sizes (default: 64)",
    )
    args = parser.parse_args()

    base = args.url.rstrip("/")

    try:
        r = requests.get(f"{base}/v2/health/ready", timeout=5)
        if r.status_code != 200:
            print(f"Server not ready: {r.status_code}", file=sys.stderr)
            sys.exit(1)
    except requests.RequestException as e:
        print(f"Cannot reach Triton at {base}: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Saturation benchmark: batch sizes {BATCH_SIZES}, {args.duration}s each, "
        f"{args.workers} workers, max {args.max_inflight} images in flight.\n"
    )
    print(f"{'batch_size':<12} {'images':<10} {'wall_sec':<12} {'throughput (img/s)':<20}")
    print("-" * 54)

    results = []
    for i, batch_size in enumerate(BATCH_SIZES):
        if i > 0:
            time.sleep(5)
        try:
            images_in_window, window_sec, throughput = run_saturation_test(
                base,
                args.model,
                batch_size,
                args.duration,
                args.workers,
                args.max_inflight,
            )
            results.append((batch_size, images_in_window, window_sec, throughput))
            print(f"{batch_size:<12} {images_in_window:<10} {window_sec:<12.2f} {throughput:<20.2f}")
        except requests.RequestException as e:
            print(f"batch_size={batch_size} FAILED: {e}", file=sys.stderr)
            sys.exit(1)

    with open("throughput.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "total_images", "wall_sec", "throughput_img_per_sec"])
        writer.writerows(results)

    print("-" * 54)
    print("Done. Results saved to throughput.csv")


if __name__ == "__main__":
    main()
