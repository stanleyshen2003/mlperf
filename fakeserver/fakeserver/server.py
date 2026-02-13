"""
Fake model server for load testing: ResNet, YOLO, BERT.
No GPU required: each endpoint sleeps 0.5s and returns a fake value.
"""

import argparse
import time

from flask import Flask, request, jsonify

app = Flask(__name__)

# Simulated inference delay in seconds (no real model)
INFERENCE_SLEEP_SEC = 0.5


@app.route("/resnet/", methods=["POST", "GET"])
def resnet():
    """
    Fake ResNet classification endpoint.
    Accepts optional JSON body; sleeps 0.5s and returns a fake class id and score.
    """
    time.sleep(INFERENCE_SLEEP_SEC)
    # Fake ImageNet-style response: class index and score
    return jsonify({
        "model": "resnet",
        "class_id": 42,
        "class_label": "fake_class",
        "score": 0.99,
    })


@app.route("/yolo/", methods=["POST", "GET"])
def yolo():
    """
    Fake YOLO detection endpoint.
    Accepts optional JSON body; sleeps 0.5s and returns fake bounding boxes.
    """
    time.sleep(INFERENCE_SLEEP_SEC)
    # Fake COCO-style detections: list of [ymin, xmin, ymax, xmax, score, category_id]
    return jsonify({
        "model": "yolo",
        "detections": [
            {"bbox": [0.1, 0.2, 0.5, 0.6], "score": 0.95, "category_id": 1},
            {"bbox": [0.3, 0.1, 0.7, 0.4], "score": 0.88, "category_id": 2},
        ],
    })


@app.route("/bert/", methods=["POST", "GET"])
def bert():
    """
    Fake BERT (e.g. SQuAD) endpoint.
    Accepts optional JSON body; sleeps 0.5s and returns a fake span.
    """
    time.sleep(INFERENCE_SLEEP_SEC)
    # Fake SQuAD-style span
    return jsonify({
        "model": "bert",
        "start": 0,
        "end": 5,
        "answer": "fake answer",
        "score": 0.97,
    })


@app.route("/health/", methods=["GET"])
def health():
    """Health check for load balancers or scripts."""
    return jsonify({"status": "ok", "models": ["resnet", "yolo", "bert"]})


def main():
    parser = argparse.ArgumentParser(description="Fake model server (ResNet, YOLO, BERT)")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Simulated inference sleep in seconds (default: 0.5)",
    )
    args = parser.parse_args()

    global INFERENCE_SLEEP_SEC
    INFERENCE_SLEEP_SEC = args.sleep

    print(f"Fake server: http://{args.host}:{args.port}")
    print("  /resnet/  - fake classification")
    print("  /yolo/    - fake detection")
    print("  /bert/    - fake QA span")
    print("  /health/  - health check")
    print(f"  (sleep {INFERENCE_SLEEP_SEC}s per request)")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
