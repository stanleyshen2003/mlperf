#!/usr/bin/env python3
"""
Simple test for Triton YOLO ONNX model.
Sends one inference request and checks output shape.

Usage:
  # With port-forward: kubectl port-forward svc/triton-yolo 8000:8000
  python serving/scripts/test_triton_inference.py
  python serving/scripts/test_triton_inference.py --url http://localhost:8000

Or from inside the cluster (e.g. in a pod):
  python test_triton_inference.py --url http://triton-yolo.default.svc.cluster.local:8000
"""
import argparse
import base64
import json
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Test Triton YOLO inference")
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

    # Input: [1, 3, 640, 640] FP32 (NCHW), fill with zeros for a minimal test
    shape = [1, 3, 640, 640]
    data = np.zeros(shape, dtype=np.float32)
    # Triton REST expects "data" as a JSON array of numbers (not base64 for this endpoint)
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

    try:
        r = requests.post(
            f"{base}/v2/models/{args.model}/infer",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"Inference request failed: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print(e.response.text[:500], file=sys.stderr)
        sys.exit(1)

    out = r.json()
    outputs = {o["name"]: o for o in out.get("outputs", [])}
    if "output0" not in outputs:
        print("No output0 in response", file=sys.stderr)
        sys.exit(1)

    o = outputs["output0"]
    out_shape = o.get("shape", [])
    out_data = o.get("data")
    if out_data:
        if isinstance(out_data, list):
            arr = np.array(out_data, dtype=np.float32)
        else:
            decoded = base64.b64decode(out_data)
            arr = np.frombuffer(decoded, dtype=np.float32)
        print(f"Output shape: {out_shape}")
        print(f"Output size: {len(arr)} elements")
        print(f"Output min/max: {arr.min():.4f} / {arr.max():.4f}")
    else:
        print(f"Output shape: {out_shape}")

    print("OK: inference succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
