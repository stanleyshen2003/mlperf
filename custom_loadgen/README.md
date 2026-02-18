# Custom LoadGen — Priority-Aware Scheduler Load Generator

MLPerf LoadGen (Server scenario, Poisson arrivals) driving traffic to a **custom scheduler via gRPC**. Each request carries `(tier, deadline_ns, model_name)` and a random float32 input tensor. One LoadGen instance per request class `(tier, slo)`, run concurrently in threads.

## Dependencies

- **mlperf_loadgen** — build and install from repo: `cd loadgen && pip install .` or `uv pip install -e loadgen`
- **grpcio**, **grpcio-tools**, **numpy** — in project `pyproject.toml`; use `uv sync` from repo root

## Regenerate gRPC stubs

From repo root (with `uv`):

```bash
cd custom_loadgen && uv run bash generate_proto.sh
```

## How to run

From repo root:

```bash
# Ensure scheduler gRPC server is running on localhost:50051 (or use --scheduler-url)

# Light load, balanced (outputs under result/)
uv run python custom_loadgen/loadgen_runner.py \
  --scheduler-url localhost:50051 \
  --total-qps 20 \
  --workload balanced \
  --duration-ms 60000 \
  --metrics-csv result/balanced_20qps.csv

# Heavy load, urgent-heavy
uv run python custom_loadgen/loadgen_runner.py \
  --scheduler-url localhost:50051 \
  --total-qps 70 \
  --workload urgent_heavy \
  --duration-ms 60000 \
  --metrics-csv result/urgent_heavy_70qps.csv
```

**CLI arguments:**

| Argument            | Default               | Description                          |
|---------------------|----------------------|--------------------------------------|
| `--scheduler-url`   | `localhost:50051`    | Scheduler gRPC endpoint              |
| `--total-qps`       | `30.0`               | Total aggregate QPS                  |
| `--duration-ms`     | `60000`              | Test duration (ms)                   |
| `--workload`        | `balanced`           | `balanced`, `urgent_heavy`, `batch_heavy` |
| `--min-query-count` | `100`                | LoadGen min query count              |
| `--output-dir`      | `result/loadgen_logs`| LoadGen log root                     |
| `--metrics-csv`     | `result/metrics.csv` | Per-request output CSV               |

Outputs: per-request CSV and a summary table by `(tier, slo)` with count, p50/p95/p99 ms, and SLO attainment %.
