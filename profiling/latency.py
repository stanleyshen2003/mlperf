#!/usr/bin/env python3
"""
Latency benchmark for Triton ONNX models (YOLO, ViT, ResNet, BERT).
Sends batch requests from batch size 1 to 16, 100 times each, strictly sequential
(all previous requests are processed before sending the next). Records average
and std of request time in ms, and dumps all latency records to <base>_dump.csv.
Uses Triton gRPC client (tritonclient.grpc) for lower latency than HTTP.

Usage:
  # Port-forward gRPC (Triton gRPC is typically on 8001), then from repo root:
  kubectl port-forward svc/triton-yolo 8001:8001   # or svc for model-repo2 (BERT)
  uv run python profiling/latency.py
  uv run python profiling/latency.py --model bert_onnx --url localhost:8001 --output profiling/result/bert_onnx_latency.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient

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


def _parse_grpc_url(url: str) -> str:
    """Return host:port for gRPC. If url is http(s)://host:8000, use port 8001 (Triton gRPC)."""
    s = url.strip().rstrip("/")
    if s.startswith("http://"):
        s = s[7:]
    elif s.startswith("https://"):
        s = s[8:]
    if ":" in s:
        host, port = s.rsplit(":", 1)
        if port == "8000":
            return f"{host}:8001"
        return f"{host}:{port}"
    return f"{s}:8001"


def _build_grpc_inputs(model: str, batch_size: int) -> tuple[list, list]:
    """Build gRPC InferInput and InferRequestedOutput for the given model and batch size."""
    cfg = MODEL_IO[model]
    if isinstance(cfg, dict) and cfg.get("kind") == "bert":
        seq_len = cfg["seq_len"]
        output_name = cfg["output"]
        shape = [batch_size, seq_len]
        input_ids = np.random.randint(0, 30000, size=shape, dtype=np.int64)
        attention_mask = np.ones(shape, dtype=np.int64)
        token_type_ids = np.zeros(shape, dtype=np.int64)
        inputs = [
            grpcclient.InferInput("input_ids", shape, "INT64"),
            grpcclient.InferInput("attention_mask", shape, "INT64"),
            grpcclient.InferInput("token_type_ids", shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        inputs[2].set_data_from_numpy(token_type_ids)
        outputs = [grpcclient.InferRequestedOutput(output_name)]
        return inputs, outputs
    # Vision: single FP32 input
    input_name, output_name, (c, h, w) = cfg
    shape = [batch_size, c, h, w]
    data = np.random.randn(*shape).astype(np.float32)
    inputs = [grpcclient.InferInput(input_name, shape, "FP32")]
    inputs[0].set_data_from_numpy(data)
    outputs = [grpcclient.InferRequestedOutput(output_name)]
    return inputs, outputs


def run_batch_requests(
    grpc_url: str,
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
    times_ms = []
    with grpcclient.InferenceServerClient(url=grpc_url) as client:
        for _ in range(num_requests):
            inputs, outputs = _build_grpc_inputs(model, batch_size)
            start = time.perf_counter()
            client.infer(model_name=model, inputs=inputs, outputs=outputs, client_timeout=timeout)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)
    return times_ms


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for Triton ONNX models (batch 1-16, sequential)")
    parser.add_argument(
        "--url",
        default="localhost:8001",
        help="Triton gRPC endpoint host:port (default: localhost:8001). If you pass http://host:8000, port 8001 is used.",
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

    grpc_url = _parse_grpc_url(args.url)
    out_path = Path(args.output) if args.output else RESULT_DIR / f"{args.model}_latency.csv"
    dump_path = out_path.parent / f"{out_path.stem}_dump.csv"

    try:
        with grpcclient.InferenceServerClient(url=grpc_url) as client:
            if not client.is_server_ready():
                print("Server not ready", file=sys.stderr)
                sys.exit(1)
    except Exception as e:
        print(f"Cannot reach Triton at {grpc_url}: {e}", file=sys.stderr)
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
                grpc_url, args.model, batch_size, args.requests_per_batch
            )
        except Exception as e:
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
