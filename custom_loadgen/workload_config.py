# Tiers
TIERS = [1, 2, 3]

# SLO names and latency targets in nanoseconds
SLO_LATENCY_NS = {
    "urgent": 100_000_000,  # 100ms
    "normal": 500_000_000,  # 500ms
    "relaxed": 2_000_000_000,  # 2000ms
}

# 9 request classes
REQUEST_CLASSES = [
    (1, "urgent"),
    (1, "normal"),
    (1, "relaxed"),
    (2, "urgent"),
    (2, "normal"),
    (2, "relaxed"),
    (3, "urgent"),
    (3, "normal"),
    (3, "relaxed"),
]

# Model assignment per SLO (name, probability)
MODEL_FOR_SLO = {
    "urgent": [("resnet50", 1.0)],
    "normal": [("resnet50", 1.0)],
    "relaxed": [("resnet50", 1.0)],
}

# Input tensor shape (same for both models in this setup)
INPUT_SHAPE = [1, 3, 224, 224]  # 150528 floats

# Workload distributions: fraction of total QPS per class
# Order: (1U, 1N, 1R, 2U, 2N, 2R, 3U, 3N, 3R)
WORKLOAD_DISTRIBUTIONS = {
    "balanced": {
        (1, "urgent"): 0.05,
        (1, "normal"): 0.10,
        (1, "relaxed"): 0.15,
        (2, "urgent"): 0.10,
        (2, "normal"): 0.15,
        (2, "relaxed"): 0.10,
        (3, "urgent"): 0.10,
        (3, "normal"): 0.15,
        (3, "relaxed"): 0.10,
    },
    "urgent_heavy": {
        (1, "urgent"): 0.20,
        (1, "normal"): 0.10,
        (1, "relaxed"): 0.05,
        (2, "urgent"): 0.15,
        (2, "normal"): 0.10,
        (2, "relaxed"): 0.05,
        (3, "urgent"): 0.15,
        (3, "normal"): 0.10,
        (3, "relaxed"): 0.10,
    },
    "batch_heavy": {
        (1, "urgent"): 0.05,
        (1, "normal"): 0.05,
        (1, "relaxed"): 0.20,
        (2, "urgent"): 0.05,
        (2, "normal"): 0.10,
        (2, "relaxed"): 0.20,
        (3, "urgent"): 0.05,
        (3, "normal"): 0.10,
        (3, "relaxed"): 0.20,
    },
}
