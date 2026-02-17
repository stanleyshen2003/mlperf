# YOLO ONNX with Triton on Kubernetes

This directory contains everything to **create a YOLOv11 ONNX model** (same architecture as the [MLPerf YOLO benchmark](../vision/classification_and_detection/yolo/README.md)) and run it with **NVIDIA Triton Inference Server** inside Kubernetes, using a **PVC** for the model repository and configurations.

## Overview

- **Model**: YOLOv11 (Ultralytics), same architecture as MLPerf YOLO. We use Ultralytics pretrained weights (e.g. `yolo11n.pt`) and export to ONNX—no MLPerf/Cloudflare download required.
- **Runtime**: Triton Inference Server with ONNX backend.
- **Storage**: A PersistentVolumeClaim (`triton-model-repo`) holds the model repo (e.g. `yolo_onnx/1/model.onnx` and `yolo_onnx/config.pbtxt`).
- **Flow**: Upload the model from your local `serving/model-repo/` into the PVC via an upload pod, then deploy Triton to serve from that PVC. (Optional: use the download Job to create the model inside the cluster.)

## Prerequisites

- Kubernetes cluster (you already have this).
- `kubectl` configured.
- For GPU acceleration: nodes with NVIDIA GPU and [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) (optional; Triton can run on CPU).

## Quick start

### 1. Create namespace (optional)

```bash
kubectl create namespace triton-yolo
kubectl config set-context --current --namespace=triton-yolo
```

### 2. Apply manifests and upload model from local files

```bash
# From repo root: apply PVC, ConfigMap, Triton Deployment and Service (no Job)
kubectl apply -k serving/k8s

# Upload local model repo into the PVC
kubectl apply -f serving/k8s/upload-pod.yaml
kubectl wait --for=condition=Ready pod/upload-yolo-model --timeout=60s
kubectl cp serving/model-repo/. upload-yolo-model:/models/
kubectl delete pod upload-yolo-model

# Restart Triton so it loads the model
kubectl rollout restart deployment/triton-yolo
```

### 3. Check Triton is ready

```bash
kubectl get pods -l app=triton-yolo
kubectl logs -l app=triton-yolo -f
# When ready:
kubectl port-forward svc/triton-yolo 8000:8000 8001:8001
# Then: curl http://localhost:8000/v2/health/ready
# And:  curl http://localhost:8000/v2/models/yolo_onnx
```

## Stable Diffusion XL (model-repo5)

Stable Diffusion XL from the [text_to_image](../text_to_image/) benchmark, exported to ONNX and laid out for Triton (one ONNX model per pipeline component).

From repo root with **uv**:

```bash
uv sync --extra sdxl-export
uv run serving/scripts/export_stable_diffusion_xl_onnx.py
```

Output: `serving/model-repo5/` with `text_encoder_onnx`, `text_encoder_2_onnx`, `unet_onnx`, `vae_decoder_onnx` (each with `1/model.onnx` and `config.pbtxt`). Optionally pass `--model-id` and `--output-repo`.

## Directory layout

```
serving/
├── README.md                 # This file
├── k8s/
│   ├── pvc.yaml              # PVC for model repo
│   ├── configmap.yaml        # setup_triton_repo.py + config.pbtxt
│   ├── upload-pod.yaml       # One-off pod to kubectl cp local model-repo into PVC
│   ├── download-job.yaml     # Optional: Job to create model in-cluster (not used by default)
│   └── triton-deployment.yaml # Triton Deployment + Service
├── model-repo/
│   └── yolo_onnx/
│       └── config.pbtxt      # Triton model config (reference)
├── model-repo5/              # Stable Diffusion XL ONNX (Triton style), created by export script
│   └── <component>_onnx/
│       ├── config.pbtxt
│       └── 1/
│           └── model.onnx
└── scripts/
    ├── run_download_with_uv.py       # Local: run from repo root with uv (create ONNX + setup repo)
    ├── export_yolo_onnx.py            # Export YOLOv11 from Ultralytics to ONNX
    ├── export_stable_diffusion_xl_onnx.py  # Export SDXL to ONNX for Triton (use with uv)
    ├── download_model.sh              # Local: optional mlcr path for official MLPerf model
    └── setup_triton_repo.py           # Optional: from mlcr download dir to Triton repo
```

## PVC and storage class

- Default `pvc.yaml` requests **2Gi** and **ReadWriteMany** (e.g. for NFS/EFS so multiple pods could share it). If your cluster only supports **ReadWriteOnce**, change `accessModes` to `ReadWriteOnce` and keep a single Triton replica.
- To use a specific storage class (e.g. NFS), set `storageClassName` in `pvc.yaml`.

## GPU

To use GPU with Triton, uncomment the `nvidia.com/gpu` request/limit in `serving/k8s/triton-deployment.yaml` and ensure the NVIDIA device plugin is installed in the cluster.

## Local run (without Kubernetes)

### Using uv from the repo root (recommended)

From the **outer folder** (repository root) use `uv` so the same venv and deps are used:

```bash
# From repo root (e.g. /path/to/mlperf)
uv sync --extra yolo-serving
uv run --extra yolo-serving python serving/scripts/run_download_with_uv.py
```

This creates a YOLOv11 ONNX model (Ultralytics, same architecture as MLPerf YOLO) and writes the Triton model repo to **`serving/model-repo/`**:
- `serving/model-repo/yolo_onnx/1/model.onnx`
- `serving/model-repo/yolo_onnx/config.pbtxt`

Then run Triton with:
```bash
tritonserver --model-repository=serving/model-repo
```

### Using pip

```bash
cd serving/scripts
pip install ultralytics opencv-python-headless
python export_yolo_onnx.py --output-dir /path/to/model-repo/yolo_onnx/1 --variant n
# Copy config.pbtxt to /path/to/model-repo/yolo_onnx/ then run Triton with --model-repository=/path/to/model-repo
```

## Triton API

- HTTP: `http://<service>:8000`
- gRPC: `<service>:8001`
- Model name: `yolo_onnx`
- Input: `images`, shape `[batch, 3, 640, 640]`, FP32, NCHW.
- Output: `output0`, shape `[batch, 84, 8400]` (standard YOLO raw outputs; client should run NMS/post-processing).
