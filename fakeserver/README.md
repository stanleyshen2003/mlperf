# Fake Model Server

Fake inference server for load testing **without GPU**: exposes three endpoints (ResNet, YOLO, BERT). Each request sleeps 0.5 seconds and returns a fake response.

## Setup (uv)

From this folder (`fakeserver/`):

```bash
# Create venv and install dependencies (uv uses .venv here)
uv sync
```

## Run

From the `fakeserver/` directory:

```bash
uv run python run.py
```

Or with options:

```bash
uv run python run.py --host 127.0.0.1 --port 8000 --sleep 0.5
```

Alternatively: `uv run python -m fakeserver.server [options]`

**Makefile (start/stop in background):**

```bash
make start          # start on port 8000, PID in .server.pid
make start PORT=9000
make stop           # stop server
make restart        # stop then start
make status         # show if running and hit /health/
```

Server will listen on `http://127.0.0.1:8000` by default.

## Endpoints

| Path       | Method | Description                    |
|-----------|--------|--------------------------------|
| `/resnet/` | POST, GET | Fake image classification (class_id, score) |
| `/yolo/`    | POST, GET | Fake object detection (fake boxes)         |
| `/bert/`    | POST, GET | Fake QA span (start, end, answer)          |
| `/health/`  | GET    | Health check; returns `{"status": "ok", "models": ["resnet", "yolo", "bert"]}` |

Each model endpoint sleeps **0.5s** (configurable via `--sleep`) then returns a fixed fake payload. No real models or GPU required.

## Mixed workload (MLPerf LoadGen)

The **official MLPerf LoadGen** is used to generate a mixture of ResNet, YOLO, and BERT traffic against the fake server.

**1. Build and install the loadgen** (from the inference repo root):

```bash
cd loadgen
pip install .   # or: uv pip install -e .
cd ../fakeserver
```

Use the same Python/venv when running the workload (e.g. `uv run python loadgen_workload.py` after `uv pip install -e ../loadgen` from fakeserver, or install loadgen into fakeserver’s venv).

**2. Start the fake server** (e.g. `make start` or `uv run python run.py`).

**3. Run the LoadGen mixed workload:**

```bash
# Default: Server scenario, 10s, target 10 QPS, 1:1:1 sample mix
uv run python loadgen_workload.py

# Custom scenario, duration, QPS, and mix (resnet,yolo,bert ratio)
uv run python loadgen_workload.py --scenario Server --duration-ms 15000 --target-qps 12 --mix 2,1,1

# Offline scenario
uv run python loadgen_workload.py --scenario Offline --total-samples 5000 --mix 1,1,1
```

**Makefile:** `make loadgen-workload` runs the default LoadGen workload (server must be running).

**Changeable QPS (time-varying):** use `--schedule` to run LoadGen in phases with different target QPS. Example: 0–5s @ 3 QPS, 5–10s @ 10 QPS, 10–15s @ 20 QPS:

```bash
uv run python loadgen_workload.py --schedule "0-5:3,5-10:10,10-15:20" --workload-log workload_schedule.csv
```

**Makefile:** `make workload-schedule` runs the above (server must be running).

Format: `start-end:qps,...` (times in seconds). Options: `--url`, `--scenario`, `--duration-ms`, `--target-qps`, `--min-query-count`, `--mix`, `--total-samples`, `--timeout`, `--output-dir`, `--workload-log`, `--schedule`.

**See actual model inference workload (per-request):** use `--workload-log PATH` to write a CSV with each request’s timestamp, model (resnet/yolo/bert), and latency, e.g.:

```bash
uv run python loadgen_workload.py --workload-log workload.csv --duration-ms 5000
```

Then open `workload.csv`: columns `timestamp_issued`, `response_id`, `sample_index`, `model`, `tag`, `latency_ns`, `latency_ms`. Each request gets a random `tag` (1–3). You can filter/sort by `model` or `tag` to see traffic per endpoint or per tag.

## Example

```bash
curl -s http://127.0.0.1:8000/resnet/
curl -s http://127.0.0.1:8000/yolo/
curl -s http://127.0.0.1:8000/bert/
curl -s http://127.0.0.1:8000/health/
```
