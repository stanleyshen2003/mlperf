# Profiling

Performance profiling and benchmarking scripts for Triton-served models.

- **latency.py** – Latency benchmark: batch sizes 1–16, 100 sequential requests each. Writes `profiling/result/latency.csv` (avg, std in ms) and `profiling/result/latency_dump.csv` (all samples).
- **throughput.py** – Throughput (saturation) benchmark for YOLO, batch sizes 1–16. Writes `profiling/result/throughput.csv`.
- **benchmark.py** – Throughput (saturation) benchmark for YOLO ONNX (batch 1,2,4,8). Writes `throughput.csv`.
- **vit_benchmark.py** – Throughput benchmark for ViT ONNX. Writes `vit_throughput.csv` (or `--output`).
- **batch_testing.py** – Batch-size latency test for YOLO (sequential, batch 1–8). Writes `batch_test_results.csv` (or `--output`).

All results from `latency.py` and `throughput.py` are stored under **profiling/result/*.csv**.

**Run port-forward first**, then from repo root use `uv run python`:

```bash
# Port-forward first (YOLO or ViT service as needed)
kubectl port-forward svc/triton-yolo 8000:8000
# or: kubectl port-forward svc/triton-repo3 8000:8000

uv run python profiling/latency.py
uv run python profiling/throughput.py
uv run python profiling/benchmark.py
uv run python profiling/batch_testing.py
uv run python profiling/vit_benchmark.py
```
