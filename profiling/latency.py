#!/usr/bin/env python3
"""
Latency benchmark for Triton ONNX models (YOLO, ViT, ResNet, BERT).
Sends batch requests from batch size 1 to 16, 100 times each, strictly sequential
(all previous requests are processed before sending the next). Records average
and std of request time in ms, and dumps all latency records to <base>_dump.csv.

Usage:
  # Port-forward first, then from repo root:
  kubectl port-forward svc/triton-yolo 8000:8000   # or svc for model-repo2 (BERT)
  uv run python profiling/latency.py
  uv run python profiling/latency.py --model bert_onnx --url http://localhost:8000 --output profiling/result/bert_onnx_latency.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import requests

# Batch sizes to test
BATCH_SIZES = list(range(1, 17))
DEFAULT_REQUESTS_PER_BATCH = 100
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


def _build_payload(model: str, batch_size: int) -> tuple[dict, str]:
    """Build inference payload for the given model and batch size. Returns (payload, output_name)."""
    cfg = MODEL_IO[model]
    if isinstance(cfg, dict) and cfg.get("kind") == "bert":
        seq_len = cfg["seq_len"]
        output_name = cfg["output"]
        shape = [batch_size, seq_len]
        # BERT: input_ids, attention_mask, token_type_ids (INT64). Use small token ids for input_ids.
        input_ids = np.random.randint(0, 30000, size=shape, dtype=np.int64)
        attention_mask = np.ones(shape, dtype=np.int64)
        token_type_ids = np.zeros(shape, dtype=np.int64)
        payload = {
            "inputs": [
                {"name": "input_ids", "shape": shape, "datatype": "INT64", "data": input_ids.flatten().tolist()},
                {"name": "attention_mask", "shape": shape, "datatype": "INT64", "data": attention_mask.flatten().tolist()},
                {"name": "token_type_ids", "shape": shape, "datatype": "INT64", "data": token_type_ids.flatten().tolist()},
            ],
            "outputs": [{"name": output_name}],
        }
        return payload, output_name
    # Vision: single FP32 input
    input_name, output_name, (c, h, w) = cfg
    shape = [batch_size, c, h, w]
    data = np.random.randn(*shape).astype(np.float32)
    payload = {
        "inputs": [
            {"name": input_name, "shape": shape, "datatype": "FP32", "data": data.flatten().tolist()},
        ],
        "outputs": [{"name": output_name}],
    }
    return payload, output_name


def run_batch_requests(
    base_url: str,
    model: str,
    batch_size: int,
    num_requests: int,
    timeout: int = 120,
) -> list[float]:
    """
    Send num_requests inference requests with the given batch_size, one by one.
    Each request is sent only after the previous one has completed.
    Returns list of latency in milliseconds for each request.
    """
    if model not in MODEL_IO:
        raise ValueError(f"Unknown model {model!r}; supported: {list(MODEL_IO)}")
    infer_url = f"{base_url}/v2/models/{model}/infer"
    times_ms = []

    for _ in range(num_requests):
        payload, _ = _build_payload(model, batch_size)
        start = time.perf_counter()
        r = requests.post(infer_url, json=payload, timeout=timeout)
        r.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

    return times_ms


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for Triton ONNX models (batch 1-16, sequential)")
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
        "--requests-per-batch",
        type=int,
        default=DEFAULT_REQUESTS_PER_BATCH,
        help=f"Number of requests per batch size (default: {DEFAULT_REQUESTS_PER_BATCH})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV for summary (avg, std); dump will be <stem>_dump.csv (default: result/<model>_latency.csv)",
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
    out_path = Path(args.output) if args.output else RESULT_DIR / f"{args.model}_latency.csv"
    dump_path = out_path.parent / f"{out_path.stem}_dump.csv"

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
        f"Latency benchmark: batch sizes {BATCH_SIZES[0]}..{BATCH_SIZES[-1]}, "
        f"{args.requests_per_batch} requests each (sequential)."
    )
    print(f"Summary: {out_path}, dump: {dump_path}\n")
    print(f"{'batch_size':<12} {'avg_ms':<12} {'std_ms':<12}")
    print("-" * 36)

    summary_rows = []
    dump_rows = []

    for batch_size in BATCH_SIZES:
        print(f"{batch_size:<12} ", end="", flush=True)
        try:
            times_ms = run_batch_requests(
                base, args.model, batch_size, args.requests_per_batch
            )
        except requests.RequestException as e:
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)

        arr = np.array(times_ms, dtype=np.float64)
        avg_ms = float(np.mean(arr))
        std_ms = float(np.std(arr))
        summary_rows.append((batch_size, avg_ms, std_ms))
        for t in times_ms:
            dump_rows.append((batch_size, t))
        print(f"{avg_ms:<12.2f} {std_ms:<12.2f}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "avg_ms", "std_ms"])
        writer.writerows(summary_rows)

    with open(dump_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "latency_ms"])
        writer.writerows(dump_rows)

    print("-" * 36)
    print(f"Done. Summary: {out_path}, dump: {dump_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
