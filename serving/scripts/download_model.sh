#!/usr/bin/env bash
# Download the official MLPerf YOLOv11 model and set up Triton model repository.
# Follows vision/classification_and_detection/yolo/README.md:
#   mlcr get-ml-model-yolov11,_mlc,_r2-downloader --outdirname=<Download path> -j
#
# Usage: download_model.sh <model-repo-root>
# Example: download_model.sh /models
# Requires: Python 3 with mlc-scripts (pip install mlc-scripts), and ultralytics if model is .pt

set -e

MODEL_REPO_ROOT="${1:-/models}"

if [[ -z "$1" ]]; then
  echo "Usage: $0 <model-repo-root>" >&2
  echo "  MODEL_REPO_ROOT: path where Triton model repo will be created (e.g. /models)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="$(mktemp -d)"
trap 'rm -rf "$DOWNLOAD_DIR"' EXIT

echo "Downloading official MLPerf YOLOv11 model (mlcr get-ml-model-yolov11,_mlc,_r2-downloader)..."
mlcr get-ml-model-yolov11,_mlc,_r2-downloader --outdirname="$DOWNLOAD_DIR" -j

if [[ ! -d "$DOWNLOAD_DIR" ]] || [[ -z "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ]]; then
  echo "Error: Download produced no files under $DOWNLOAD_DIR" >&2
  exit 1
fi

echo "Setting up Triton model repo at ${MODEL_REPO_ROOT}..."
python3 "${SCRIPT_DIR}/setup_triton_repo.py" \
  --download-dir "$DOWNLOAD_DIR" \
  --model-repo "$MODEL_REPO_ROOT"

CONFIG_SRC="${SCRIPT_DIR}/../model-repo/yolo_onnx/config.pbtxt"
if [[ -f "${CONFIG_SRC}" ]]; then
  mkdir -p "${MODEL_REPO_ROOT}/yolo_onnx"
  cp "${CONFIG_SRC}" "${MODEL_REPO_ROOT}/yolo_onnx/config.pbtxt"
  echo "Config copied."
fi

echo "Done. Model repo: ${MODEL_REPO_ROOT}"
ls -la "${MODEL_REPO_ROOT}/yolo_onnx/"
