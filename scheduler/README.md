# Scheduler gRPC server

Listens for inference requests from the custom_loadgen client. Currently **log-only**: prints each request’s `tier`, `deadline_ns`, `model_name`, and input shape/length (not the tensor data) so you can verify the load generator.

## Run

From repo root:

```bash
uv run python scheduler/server.py
# default: 0.0.0.0:50051

uv run python scheduler/server.py --port 50051 --host 0.0.0.0
```

Then run the load generator (in another terminal):

```bash
uv run python custom_loadgen/loadgen_runner.py --scheduler-url localhost:50051 --total-qps 10 --duration-ms 10000
```

You should see `[Infer] tier=... deadline_ns=... model_name=...` lines on the scheduler terminal.
