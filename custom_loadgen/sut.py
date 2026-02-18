"""
SUT that dispatches inference requests to the custom scheduler via gRPC.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path
import numpy as np

# Allow importing generated gRPC stubs
sys.path.insert(0, str(Path(__file__).resolve().parent / "generated"))

import grpc
import mlperf_loadgen as lg

from workload_config import (
    INPUT_SHAPE,
    MODEL_FOR_SLO,
    SLO_LATENCY_NS,
)

# Generated stubs (after generate_proto.sh)
from scheduler_pb2 import InferRequest, InferResponse
from scheduler_pb2_grpc import SchedulerServiceStub


class SchedulerSUT:
    """
    System-under-test that sends each query to the scheduler gRPC endpoint
    with (tier, deadline_ns, model_name) and a random float32 input tensor.
    """

    def __init__(self, scheduler_url: str, tier: int, slo: str, seed: int):
        self.scheduler_url = scheduler_url
        self.tier = tier
        self.slo = slo
        self.records: list = []
        self.lock = threading.Lock()
        self._channel = grpc.insecure_channel(scheduler_url)
        self._stub = SchedulerServiceStub(self._channel)
        self._lg_sut = None
        random.seed(seed)
        np.random.seed(seed)

    def issue_query(self, query_samples):
        # Copy list so LoadGen can reuse its buffers
        samples = [(s.id, s.index) for s in query_samples]

        def run():
            responses = []
            for sample_id, _index in samples:
                timestamp_issued = time.time()
                t0 = time.perf_counter()

                # Sample model from MODEL_FOR_SLO[slo]
                names = [t[0] for t in MODEL_FOR_SLO[self.slo]]
                weights = [t[1] for t in MODEL_FOR_SLO[self.slo]]
                model_name = random.choices(names, weights=weights, k=1)[0]

                # Random float32 tensor per request
                tensor = np.random.rand(*INPUT_SHAPE).astype(np.float32)
                deadline_ns = time.time_ns() + SLO_LATENCY_NS[self.slo]

                request = InferRequest(
                    tier=self.tier,
                    deadline_ns=deadline_ns,
                    model_name=model_name,
                    input_tensor=tensor.flatten().tolist(),
                    input_shape=INPUT_SHAPE,
                )

                try:
                    self._stub.Infer(request, timeout=5.0)
                except Exception:
                    pass  # Still record latency and call QuerySamplesComplete

                t1 = time.perf_counter()
                latency_ns = int((t1 - t0) * 1e9)
                slo_met = latency_ns <= SLO_LATENCY_NS[self.slo]

                with self.lock:
                    self.records.append({
                        "timestamp_issued": timestamp_issued,
                        "sample_id": sample_id,
                        "tier": self.tier,
                        "slo": self.slo,
                        "model_name": model_name,
                        "latency_ns": latency_ns,
                        "slo_met": slo_met,
                    })

                responses.append(lg.QuerySampleResponse(sample_id, 0, 0))

            lg.QuerySamplesComplete(responses)

        threading.Thread(target=run).start()

    def flush_queries(self):
        pass

    def build_sut(self):
        self._lg_sut = lg.ConstructSUT(self.issue_query, self.flush_queries)
        return self._lg_sut

    def destroy(self):
        if self._lg_sut is not None:
            lg.DestroySUT(self._lg_sut)
            self._lg_sut = None
        self._channel.close()
