#!/usr/bin/env python3
"""
Create a YOLOv11 ONNX model (same architecture as MLPerf YOLO) and set up Triton model repo.
Uses Ultralytics pretrained weights; no MLPerf/Cloudflare download required.

Run from repo root with: uv run --extra yolo-serving python serving/scripts/run_download_with_uv.py

Puts the model at serving/model-repo/yolo_onnx/1/model.onnx (config at serving/model-repo/yolo_onnx/config.pbtxt).
"""
import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    model_repo = repo_root / "serving" / "model-repo"
    version_dir = model_repo / "yolo_onnx" / "1"
    scripts_dir = repo_root / "serving" / "scripts"

    print("Creating YOLOv11 ONNX (Ultralytics, same architecture as MLPerf YOLO)...")
    r = subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "export_yolo_onnx.py"),
            "--output-dir",
            str(version_dir),
            "--variant",
            "n",
        ],
        cwd=repo_root,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)
    if not (version_dir / "model.onnx").exists():
        print("Export failed: model.onnx not created", file=sys.stderr)
        sys.exit(1)
    print(f"Done. Model repo: {model_repo}")
    print(f"  {model_repo / 'yolo_onnx' / 'config.pbtxt'}")
    print(f"  {version_dir / 'model.onnx'}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
