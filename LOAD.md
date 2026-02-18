# Load Generator Implementation Instructions

## Overview

Implement a **priority-aware load generator** for the GPU inference scheduling research project. It uses **MLPerf LoadGen** (Server scenario, Poisson arrivals) to drive traffic to a **custom scheduler via gRPC**. Each request carries `(tier, deadline_ns, model_name)` metadata plus a random input tensor.

> **Note:** Triton gRPC integration comes later. For now, the target is the custom scheduler's gRPC endpoint defined below.

---

## Project Context

- **9 request classes:** `(tier, slo)` where Tier ∈ {1, 2, 3} and SLO ∈ {Urgent=100ms, Normal=500ms, Relaxed=2000ms}
- **Models:** `resnet50` and `vit`
  - Urgent requests → ResNet-50 only
  - Normal/Relaxed requests → 70% ResNet-50, 30% ViT
- **Traffic pattern:** Poisson arrivals (MLPerf Server scenario), one independent LoadGen instance per class, all run concurrently via threads
- **Transport:** Custom gRPC scheduler endpoint defined by `scheduler.proto`

---

## File Structure to Create

```
custom_loadgen/
├── proto/
│   └── scheduler.proto         # gRPC interface definition
├── generated/                  # Output of protoc codegen (do not edit manually)
│   ├── scheduler_pb2.py
│   └── scheduler_pb2_grpc.py
├── loadgen_runner.py            # Main entry point
├── sut.py                       # SUT: gRPC dispatch to scheduler
├── workload_config.py           # Request class definitions, mix distributions, SLO targets
├── qsl.py                       # QSL factory
├── metrics.py                   # Per-request logging and CSV output
├── generate_proto.sh            # One-liner to regenerate gRPC stubs
└── README.md                    # How to run
```

---

## Dependencies

```
mlperf_loadgen        # built and installed from mlcommons/inference repo
grpcio
grpcio-tools          # for protoc codegen
numpy
```

---

## Step 1: Define the Proto — `proto/scheduler.proto`

```protobuf
syntax = "proto3";

package scheduler;

// A single inference request sent from the load generator to the scheduler.
message InferRequest {
  // Priority tier: 1 = VIP, 2 = Premium, 3 = Free
  int32 tier = 1;

  // Absolute deadline as Unix timestamp in nanoseconds (Unix epoch).
  // e.g. 1740888024000000000 == 2026-03-02 18:20:24 UTC
  // Computed as: time.time_ns() + slo_latency_ns at the moment of dispatch.
  // The scheduler compares this against time.time_ns() to check expiry.
  int64 deadline_ns = 2;

  // Model to run: "resnet50" or "vit"
  string model_name = 3;

  // Flattened float32 input tensor.
  // resnet50: shape [1, 3, 224, 224] → 150528 floats
  // vit:      shape [1, 3, 224, 224] → 150528 floats
  repeated float input_tensor = 4;

  // Shape of the input tensor (e.g. [1, 3, 224, 224])
  repeated int32 input_shape = 5;
}

// Response from the scheduler after inference completes.
message InferResponse {
  // Echo back the request's deadline_ns so the client can compute latency.
  int64 deadline_ns = 1;

  // Whether the request was processed successfully.
  bool success = 2;

  // Optional: human-readable status or error message.
  string message = 3;
}

service SchedulerService {
  // Submit one inference request to the scheduler.
  rpc Infer(InferRequest) returns (InferResponse);
}
```

**To regenerate stubs after editing the proto:**

```bash
# generate_proto.sh
python -m grpc_tools.protoc \
  -I proto/ \
  --python_out=generated/ \
  --grpc_python_out=generated/ \
  proto/scheduler.proto
```

Add an empty `generated/__init__.py` after running this.

---

## Step 2: `workload_config.py`

```python
# Tiers
TIERS = [1, 2, 3]

# SLO names and latency targets in nanoseconds
SLO_LATENCY_NS = {
    "urgent":  100_000_000,    # 100ms
    "normal":  500_000_000,    # 500ms
    "relaxed": 2_000_000_000,  # 2000ms
}

# 9 request classes
REQUEST_CLASSES = [
    (1, "urgent"), (1, "normal"), (1, "relaxed"),
    (2, "urgent"), (2, "normal"), (2, "relaxed"),
    (3, "urgent"), (3, "normal"), (3, "relaxed"),
]

# Model assignment per SLO (name, probability)
MODEL_FOR_SLO = {
    "urgent":  [("resnet50", 1.0)],
    "normal":  [("resnet50", 0.7), ("vit", 0.3)],
    "relaxed": [("resnet50", 0.7), ("vit", 0.3)],
}

# Input tensor shape (same for both models in this setup)
INPUT_SHAPE = [1, 3, 224, 224]  # 150528 floats

# Workload distributions: fraction of total QPS per class
# Order: (1U, 1N, 1R, 2U, 2N, 2R, 3U, 3N, 3R)
WORKLOAD_DISTRIBUTIONS = {
    "balanced": {
        (1,"urgent"):0.05, (1,"normal"):0.10, (1,"relaxed"):0.15,
        (2,"urgent"):0.10, (2,"normal"):0.15, (2,"relaxed"):0.10,
        (3,"urgent"):0.10, (3,"normal"):0.15, (3,"relaxed"):0.10,
    },
    "urgent_heavy": {
        (1,"urgent"):0.20, (1,"normal"):0.10, (1,"relaxed"):0.05,
        (2,"urgent"):0.15, (2,"normal"):0.10, (2,"relaxed"):0.05,
        (3,"urgent"):0.15, (3,"normal"):0.10, (3,"relaxed"):0.10,
    },
    "batch_heavy": {
        (1,"urgent"):0.05, (1,"normal"):0.05, (1,"relaxed"):0.20,
        (2,"urgent"):0.05, (2,"normal"):0.10, (2,"relaxed"):0.20,
        (3,"urgent"):0.05, (3,"normal"):0.10, (3,"relaxed"):0.20,
    },
}
```

---

## Step 3: `qsl.py`

```python
import mlperf_loadgen as lg

TOTAL_SAMPLES = 1000
PERF_SAMPLE_COUNT = 500

def make_qsl() -> lg.QuerySampleLibrary:
    def load_samples(indices): pass
    def unload_samples(indices): pass
    return lg.ConstructQSL(
        TOTAL_SAMPLES, PERF_SAMPLE_COUNT,
        load_samples, unload_samples
    )
```

---

## Step 4: `sut.py`

SUT that dispatches requests to the scheduler via gRPC.

**Implementation requirements:**

1. Import from `generated/scheduler_pb2` and `generated/scheduler_pb2_grpc`
2. Create one shared `grpc.insecure_channel(scheduler_url)` and one `SchedulerServiceStub` at construction
3. In `issue_query(query_samples)`, dispatch all samples in a single background thread (mirror the example pattern: `threading.Thread(target=run).start()`)
4. For each sample in the thread:
   - Sample model name from `MODEL_FOR_SLO[slo]` using `random.choices`
   - Generate a **random float32 tensor**: `numpy.random.rand(*INPUT_SHAPE).astype(numpy.float32)`
   - Compute `deadline_ns = time.time_ns() + SLO_LATENCY_NS[slo]`
     - This produces an absolute Unix timestamp in nanoseconds, e.g. `1740888024000000000` (equivalent to `2026-03-02 18:20:24 UTC`). The scheduler can compare it directly against `time.time_ns()` to check if the deadline has passed.
   - Build `InferRequest(tier=tier, deadline_ns=deadline_ns, model_name=model_name, input_tensor=tensor.flatten().tolist(), input_shape=INPUT_SHAPE)`
   - Call `stub.Infer(request, timeout=5.0)` inside a `try/except`
   - Record `{timestamp_issued, sample_id, tier, slo, model_name, latency_ns, slo_met}` into `self.records` (protected by `self.lock`)
   - **Always** call `lg.QuerySamplesComplete([lg.QuerySampleResponse(sample_id, 0, 0)])` — even on gRPC error
5. `flush_queries` is a no-op

**Class signature:**
```python
class SchedulerSUT:
    def __init__(self, scheduler_url: str, tier: int, slo: str): ...
    def issue_query(self, query_samples): ...
    def flush_queries(self): ...
    def build_sut(self) -> lg.SystemUnderTest: ...  # calls lg.ConstructSUT(self.issue_query, self.flush_queries)
    def destroy(self): ...                           # calls lg.DestroySUT, then channel.close()
    
    records: list   # populated during run
    lock: threading.Lock
```

---

## Step 5: `metrics.py`

**Requirements:**

1. Accepts `records` list after a run completes
2. Writes CSV with columns:
   ```
   timestamp_issued, sample_id, tier, slo, model, latency_ns, latency_ms, slo_met
   ```
3. `slo_met = latency_ns <= SLO_LATENCY_NS[slo]`
4. Prints a summary table grouped by `(tier, slo)` with: `count`, `p50_ms`, `p95_ms`, `p99_ms`, `slo_attainment_%`

**Function signatures:**
```python
def write_csv(records: list, output_path: str): ...
def print_summary(records: list): ...
```

---

## Step 6: `loadgen_runner.py`

Main entry point. Runs one LoadGen thread per request class concurrently.

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--scheduler-url` | `localhost:50051` | Scheduler gRPC endpoint |
| `--total-qps` | `30.0` | Total aggregate QPS across all classes |
| `--duration-ms` | `60000` | Test duration |
| `--workload` | `balanced` | `balanced`, `urgent_heavy`, or `batch_heavy` |
| `--min-query-count` | `100` | LoadGen min query count |
| `--output-dir` | `./loadgen_logs` | Root dir for LoadGen internal logs |
| `--metrics-csv` | `./metrics.csv` | Per-request output CSV |

**Core logic:**

```python
def run_class(tier, slo, class_qps, duration_ms, min_query_count,
              scheduler_url, output_dir, all_records, all_records_lock):
    sut = SchedulerSUT(scheduler_url, tier, slo)
    lg_sut = sut.build_sut()
    qsl = make_qsl()

    settings = lg.TestSettings()
    settings.scenario = lg.TestScenario.Server
    settings.mode = lg.TestMode.PerformanceOnly
    settings.server_target_qps = class_qps
    settings.server_target_latency_ns = SLO_LATENCY_NS[slo]
    settings.min_duration_ms = duration_ms
    settings.min_query_count = max(1, min_query_count)

    log_output = lg.LogOutputSettings()
    log_output.outdir = str(Path(output_dir) / f"tier{tier}_{slo}")
    log_output.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output
    Path(log_output.outdir).mkdir(parents=True, exist_ok=True)

    lg.StartTestWithLogSettings(lg_sut, qsl, settings, log_settings)

    sut.destroy()
    lg.DestroyQSL(qsl)

    with all_records_lock:
        all_records.extend(sut.records)


def main():
    # 1. Parse args
    # 2. Load WORKLOAD_DISTRIBUTIONS[args.workload]
    # 3. For each (tier, slo) in REQUEST_CLASSES:
    #      class_qps = total_qps * distribution[(tier, slo)]
    #      Skip if class_qps == 0
    # 4. Launch one thread per class via threading.Thread(target=run_class, ...)
    # 5. Join all threads
    # 6. Call write_csv and print_summary
```

---

## Key Implementation Notes for Cursor

1. **Proto import path:** In `sut.py`, add `sys.path.insert(0, str(Path(__file__).parent / "generated"))` before importing `scheduler_pb2`.

2. **`lg.QuerySamplesComplete` must always be called** for every issued sample, even on gRPC timeout or error — otherwise LoadGen hangs indefinitely.

3. **Poisson timing is handled by LoadGen** in Server scenario. Do not add any `time.sleep()` in `issue_query`.

4. **Each class needs its own LoadGen log subdirectory** (e.g. `loadgen_logs/tier1_urgent/`). LoadGen will silently corrupt logs if two instances share the same directory.

5. **Random tensor generation:** Use `numpy.random.rand(*INPUT_SHAPE).astype(numpy.float32)` — generate a fresh tensor per request, not a shared one. `.flatten().tolist()` converts it for the proto `repeated float` field.

6. **Thread safety:** Each `SchedulerSUT` instance has its own `records` list. Only merge into `all_records` after `lg.StartTestWithLogSettings` returns (the run is complete by then, so no lock needed on `sut.records` at that point).

7. **Skip zero-QPS classes:** If `class_qps < 0.01`, skip that class to avoid passing near-zero QPS to LoadGen.

8. **gRPC channel reuse:** One channel per `SchedulerSUT` instance is fine — each class runs in its own thread and has its own stub.

---

## Example Run Commands

```bash
# Regenerate gRPC stubs first
bash generate_proto.sh

# Light load, balanced
python loadgen_runner.py \
  --scheduler-url localhost:50051 \
  --total-qps 20 \
  --workload balanced \
  --duration-ms 60000 \
  --metrics-csv results/balanced_20qps.csv

# Heavy load, urgent-heavy
python loadgen_runner.py \
  --scheduler-url localhost:50051 \
  --total-qps 70 \
  --workload urgent_heavy \
  --duration-ms 60000 \
  --metrics-csv results/urgent_heavy_70qps.csv
```

---

## What Is NOT in Scope for This Implementation

- Triton gRPC integration — will be added in a later iteration by replacing `sut.py`
- Burst/spike traffic (Experiment 5) — implement separately
- Actual model inference — the scheduler receives real tensors but dummy values are fine
- Preemption logic — handled by the scheduler, not the load generator