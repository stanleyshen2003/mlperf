#!/usr/bin/env python3
"""
Export YOLOv11 to ONNX format for Triton Inference Server.
Uses Ultralytics to download the official YOLOv11 model and export to ONNX.
Compatible with MLPerf YOLO benchmark (vision/classification_and_detection/yolo).
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv11 to ONNX for Triton")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write model.onnx (e.g. model-repo/yolo_onnx/1)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv11 variant: n(nano), s(small), m(medium), l(large), x(extra-large)",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model_name = f"yolo11{args.variant}.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "model.onnx"

    print("Exporting to ONNX (opset 21 for Triton compatibility)...")
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        dynamic=True,
        simplify=True,
        opset=21,
    )
    exported_path = Path(exported_path)
    if not exported_path.is_absolute():
        exported_path = Path.cwd() / exported_path
    if exported_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if exported_path.resolve() != out_path.resolve():
            exported_path.rename(out_path)
        print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
