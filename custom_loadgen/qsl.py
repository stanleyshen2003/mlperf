import mlperf_loadgen as lg

TOTAL_SAMPLES = 1000
PERF_SAMPLE_COUNT = 500


def make_qsl():
    def load_samples(indices):
        pass

    def unload_samples(indices):
        pass

    return lg.ConstructQSL(
        TOTAL_SAMPLES,
        PERF_SAMPLE_COUNT,
        load_samples,
        unload_samples,
    )
