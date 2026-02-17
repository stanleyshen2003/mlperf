#!/usr/bin/env python3
"""
Set up Triton model repository from the official MLPerf YOLOv11 model
downloaded via: mlcr get-ml-model-yolov11,_mlc,_r2-downloader --outdirname=<path> -j

Finds .pt or .onnx in the download dir; if .pt, exports to ONNX for Triton.
"""
import argparse
import shutil
import sys
from pathlib import Path


def find_model(download_dir: Path):
    """Return path to first .pt or .onnx file under download_dir."""
    for ext in ("*.pt", "*.onnx"):
        for p in download_dir.rglob(ext):
            if p.is_file():
                return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Setup Triton model repo from MLPerf YOLOv11 download"
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        required=True,
        help="Directory from mlcr get-ml-model-yolov11 (--outdirname)",
    )
    parser.add_argument(
        "--model-repo",
        type=Path,
        required=True,
        help="Triton model repo root (e.g. /models)",
    )
    args = parser.parse_args()

    model_path = find_model(args.download_dir)
    if not model_path:
        print(
            f"No .pt or .onnx model found under {args.download_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    version_dir = args.model_repo / "yolo_onnx" / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = version_dir / "model.onnx"

    if model_path.suffix.lower() == ".onnx":
        print(f"Using ONNX model: {model_path}")
        shutil.copy2(model_path, out_onnx)
    else:
        print(f"Exporting PyTorch model to ONNX: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError:
            print("pip install ultralytics", file=sys.stderr)
            sys.exit(1)
        model = YOLO(str(model_path))
        exported = model.export(
            format="onnx",
            imgsz=640,
            dynamic=True,
            simplify=True,
        )
        exported = Path(exported)
        if not exported.is_absolute():
            exported = Path.cwd() / exported
        if exported.exists():
            if exported.resolve() != out_onnx.resolve():
                shutil.copy2(exported, out_onnx)
            print(f"Saved to {out_onnx}")

    if not out_onnx.exists():
        print(f"Failed to create {out_onnx}", file=sys.stderr)
        sys.exit(1)
    print(f"Triton model ready: {out_onnx}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
