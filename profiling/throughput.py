#!/usr/bin/env python3
"""
Throughput (saturation) benchmark for Triton ONNX (YOLO, ViT, ResNet, BERT).
For each batch size 1–16, runs for a fixed duration with multiple workers and
capped in-flight samples. Records throughput (samples/sec). Results under profiling/result/*.csv.
Use --model yolo_onnx, vit_onnx, resnet50_onnx, or bert_onnx (model-repo2).

Usage:
  # Port-forward first, then from repo root:
  kubectl port-forward svc/triton-yolo 8000:8000   # or svc for model-repo2 (BERT), repo3 (ViT), repo4 (ResNet)
  uv run python profiling/throughput.py
  uv run python profiling/throughput.py --model bert_onnx --duration 60 --max-inflight 64
  uv run python profiling/throughput.py --model vit_onnx --duration 60 --max-inflight 64
"""
import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import requests

# Batch sizes to test (1 to 16)
BATCH_SIZES = list(range(1, 17))
RESULT_DIR = Path(__file__).resolve().parent / "result"

# Model-specific input/output config.
# Vision: (input_name, output_name, (C, H, W)) -> single FP32 input [batch, C, H, W].
# BERT: dict with kind "bert", output name, and seq_len -> input_ids, attention_mask, token_type_ids INT64 [batch, seq_len].
MODEL_IO = {
    "yolo_onnx": ("images", "output0", (3, 640, 640)),
    "vit_onnx": ("pixel_values", "logits", (3, 224, 224)),
    "resnet50_onnx": ("input", "output", (3, 224, 224)),
    "bert_onnx": {"kind": "bert", "output": "last_hidden_state", "seq_len": 128},
}


def make_random_payload(model: str, batch_size: int) -> dict:
    """Build inference payload (new data every call). Supports vision (FP32) and BERT (INT64) models."""
    cfg = MODEL_IO[model]
    if isinstance(cfg, dict) and cfg.get("kind") == "bert":
        seq_len = cfg["seq_len"]
        output_name = cfg["output"]
        shape = [batch_size, seq_len]
        input_ids = np.random.randint(0, 30000, size=shape, dtype=np.int64)
        attention_mask = np.ones(shape, dtype=np.int64)
        token_type_ids = np.zeros(shape, dtype=np.int64)
        return {
            "inputs": [
                {"name": "input_ids", "shape": shape, "datatype": "INT64", "data": input_ids.flatten().tolist()},
                {"name": "attention_mask", "shape": shape, "datatype": "INT64", "data": attention_mask.flatten().tolist()},
                {"name": "token_type_ids", "shape": shape, "datatype": "INT64", "data": token_type_ids.flatten().tolist()},
            ],
            "outputs": [{"name": output_name}],
        }
    # Vision
    input_name, output_name, (c, h, w) = cfg
    shape = [batch_size, c, h, w]
    data = np.random.randn(*shape).astype(np.float32)
    return {
        "inputs": [
            {"name": input_name, "shape": shape, "datatype": "FP32", "data": data.flatten().tolist()},
        ],
        "outputs": [{"name": output_name}],
    }


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
    max_inflight_images so each batch size is under the same load.
    Returns (images_in_window, window_sec, throughput_images_per_sec).
    """
    if model not in MODEL_IO:
        raise ValueError(f"Unknown model {model!r}; supported: {list(MODEL_IO)}")
    infer_url = f"{base_url}/v2/models/{model}/infer"

    completions: list[tuple[float, int]] = []
    lock = threading.Lock()
    deadline = [0.0]
    sem = threading.Semaphore(max_inflight_images)

    def worker() -> None:
        while time.perf_counter() < deadline[0]:
            for _ in range(batch_size):
                sem.acquire()
            try:
                payload = make_random_payload(model, batch_size)
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

    cutoff = start + duration_sec
    images_in_window = sum(bs for (ts, bs) in completions if ts <= cutoff)
    throughput = images_in_window / duration_sec if duration_sec > 0 else 0.0
    return images_in_window, duration_sec, throughput


def main():
    parser = argparse.ArgumentParser(description="Triton throughput benchmark (batch 1-16, saturation; supports yolo_onnx, vit_onnx, resnet50_onnx, bert_onnx)")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Triton HTTP endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="vit_onnx",
        help="Model name: yolo_onnx, vit_onnx, resnet50_onnx, bert_onnx (default: vit_onnx)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Seconds to run each batch size (default: 60)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Concurrent worker threads (default: 64)",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=64,
        help="Max images in flight; keeps load equal across batch sizes (default: 64)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: result/<model>_throughput.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for input data (default: none, different each run)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    base = args.url.rstrip("/")
    out_path = Path(args.output) if args.output else RESULT_DIR / f"{args.model}_throughput.csv"

    try:
        r = requests.get(f"{base}/v2/health/ready", timeout=5)
        if r.status_code != 200:
            print(f"Server not ready: {r.status_code}", file=sys.stderr)
            sys.exit(1)
    except requests.RequestException as e:
        print(f"Cannot reach Triton at {base}: {e}", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Throughput benchmark: batch sizes 1..16, {args.duration}s each, "
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

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "total_images", "wall_sec", "throughput_img_per_sec"])
        writer.writerows(results)

    print("-" * 54)
    print(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
