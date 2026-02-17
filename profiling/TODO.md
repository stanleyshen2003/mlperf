# Profiling TODO

- [x] **latency.py** – Latency benchmark: batch sizes 1–16, 100 requests per batch size, strictly sequential (each request completes before the next is sent). Record average execution time (ms) and standard deviation (ms). Write summary to `profiling/result/<base>.csv` and all per-request latencies to `profiling/result/<base>_dump.csv`.
- [x] **throughput.py** – Throughput (saturation) benchmark based on `profiling/benchmark.py`: test batch sizes 1–16, same semantics (fixed duration, capped in-flight images). All outputs under `profiling/result/*.csv`.
