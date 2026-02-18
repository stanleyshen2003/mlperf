"""
gRPC server that implements SchedulerService. Listens for inference requests
from the custom_loadgen client and logs them (tier, deadline_ns, model_name)
so you can verify the load generator. Does not log model input tensor.

Run from repo root:
  uv run python scheduler/server.py
  uv run python scheduler/server.py --port 50051
"""

import argparse
import sys
import time
from pathlib import Path

# Use generated stubs from custom_loadgen
_repo_root = Path(__file__).resolve().parent.parent
_generated = _repo_root / "custom_loadgen" / "generated"
if _generated.exists():
    sys.path.insert(0, str(_generated))
else:
    sys.path.insert(0, str(_repo_root / "custom_loadgen"))

import grpc
from concurrent import futures

from scheduler_pb2 import InferRequest, InferResponse
from scheduler_pb2_grpc import SchedulerServiceServicer, add_SchedulerServiceServicer_to_server


class SchedulerServicer(SchedulerServiceServicer):
    """Logs each Infer request (tier, deadline_ns, model_name); does not print input tensor."""

    def Infer(self, request: InferRequest, context: grpc.ServicerContext) -> InferResponse:
        now_ns = time.time_ns()
        # Log only metadata, not input_tensor
        input_len = len(request.input_tensor)
        input_shape = list(request.input_shape) if request.input_shape else []
        print(
            f"[Infer] tier={request.tier} deadline_ns={request.deadline_ns} "
            f"model_name={request.model_name!r} "
            f"input_tensor_len={input_len} input_shape={input_shape}"
        )
        return InferResponse(
            deadline_ns=request.deadline_ns,
            success=True,
            message="OK",
        )


def main():
    parser = argparse.ArgumentParser(description="Scheduler gRPC server (log-only for loadgen testing)")
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="gRPC listen port (default: 50051)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_SchedulerServiceServicer_to_server(SchedulerServicer(), server)
    addr = f"{args.host}:{args.port}"
    server.add_insecure_port(addr)
    server.start()
    print(f"Scheduler gRPC server listening on {addr}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
