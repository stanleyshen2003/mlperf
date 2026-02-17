#!/usr/bin/env python3
"""
Example test for Triton ViT ONNX model (model-repo3).
Sends one inference request with a dummy image [1, 3, 224, 224] and prints logits summary.

Usage:
  # With port-forward: kubectl port-forward svc/triton-repo3 8000:8000
  python serving/scripts/test_triton_vit_inference.py
  python serving/scripts/test_triton_vit_inference.py --url http://localhost:8000

Or from inside the cluster:
  python serving/scripts/test_triton_vit_inference.py --url http://triton-repo3:8000
"""
import argparse
import base64
import json
import sys

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
    parser = argparse.ArgumentParser(description="Test Triton ViT inference")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Triton HTTP endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="vit_onnx",
        help="Model name (default: vit_onnx)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    args = parser.parse_args()

    base = args.url.rstrip("/")
    # Health check
    try:
        r = requests.get(f"{base}/v2/health/ready", timeout=10)
        if r.status_code != 200:
            print(f"Server not ready: {r.status_code}", file=sys.stderr)
            sys.exit(1)
    except requests.RequestException as e:
        print(f"Cannot reach Triton at {base}: {e}", file=sys.stderr)
        sys.exit(1)

    # ViT input: [batch, 3, 224, 224] FP32 (NCHW)
    shape = [args.batch_size, 3, 224, 224]
    data = np.zeros(shape, dtype=np.float32)
    # Optional: use small random values instead of zeros to avoid trivial outputs
    data = np.random.randn(*shape).astype(np.float32) * 0.1
    data_list = data.flatten().tolist()

    payload = {
        "inputs": [
            {
                "name": "pixel_values",
                "shape": shape,
                "datatype": "FP32",
                "data": data_list,
            }
        ],
        "outputs": [{"name": "logits"}],
    }

    try:
        r = requests.post(
            f"{base}/v2/models/{args.model}/infer",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"Inference request failed: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print(e.response.text[:500], file=sys.stderr)
        sys.exit(1)

    out = r.json()
    outputs = {o["name"]: o for o in out.get("outputs", [])}
    if "logits" not in outputs:
        print("No logits in response", file=sys.stderr)
        sys.exit(1)

    o = outputs["logits"]
    out_shape = o.get("shape", [])
    out_data = o.get("data")
    if out_data:
        if isinstance(out_data, list):
            arr = np.array(out_data, dtype=np.float32)
        else:
            decoded = base64.b64decode(o.get("data"))
            arr = np.frombuffer(decoded, dtype=np.float32)
        arr = arr.reshape(out_shape)
        print(f"Output shape: {out_shape}")
        print(f"Logits shape: {arr.shape}")
        print(f"Logits min/mean/max: {arr.min():.4f} / {arr.mean():.4f} / {arr.max():.4f}")
        # Top-5 class indices (for batch size 1)
        if arr.size >= 1000:
            logits_1 = arr.reshape(-1, 1000)[0]
            top5_idx = np.argsort(logits_1)[-5:][::-1]
            print(f"Top-5 class indices: {top5_idx.tolist()}")
    else:
        print(f"Output shape: {out_shape}")

    print("OK: ViT inference succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
