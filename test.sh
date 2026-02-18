#!/usr/bin/env bash
# Run latency and throughput benchmarks. For each target: port-forward -> profile (latency + throughput) -> stop port-forward.
# Usage: from repo root, ./test.sh
# Override: PORT_FORWARD_1=svc/triton-repo3 PORT_FORWARD_2=svc/triton-repo4 PORT_FORWARD_3=svc/triton-repo2 ./test.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

URL="${URL:-localhost:8001}"
LOCAL_PORT="${LOCAL_PORT:-8001}"

# One port-forward per model (each model is on a different Triton service in k8s)
# model-repo3=ViT, model-repo4=ResNet-50, model-repo2=BERT (see serving/k8s/model-repo*/triton-deployment.yaml)
PORT_FORWARD_1="${PORT_FORWARD_1:-svc/triton-repo3}"
MODELS_1=(vit_onnx)
PORT_FORWARD_2="${PORT_FORWARD_2:-svc/triton-repo4}"
MODELS_2=(resnet50_onnx)
PORT_FORWARD_3="${PORT_FORWARD_3:-svc/triton-repo2}"
MODELS_3=(bert_onnx)

run_profiles() {
  local models=("$@")
  for model in "${models[@]}"; do
    echo "========== $model =========="
    echo "--- latency ---"
    uv run python profiling/latency.py --model "$model" --url "$URL"
    echo "--- throughput ---"
    uv run python profiling/throughput.py --model "$model" --url "$URL"
    echo ""
  done
}

start_port_forward() {
  local target="$1"
  kubectl port-forward "$target" "${LOCAL_PORT}:8001" &
  PF_PID=$!
  sleep 2
  if ! kill -0 "$PF_PID" 2>/dev/null; then
    echo "Port-forward failed (is ${target} available?)." >&2
    return 1
  fi
  echo "Port-forward running for ${target} (PID $PF_PID)."
  return 0
}

stop_port_forward() {
  if [[ -n "$PF_PID" ]] && kill -0 "$PF_PID" 2>/dev/null; then
    echo "Stopping port-forward (PID $PF_PID) ..."
    kill "$PF_PID" 2>/dev/null || true
    wait "$PF_PID" 2>/dev/null || true
  fi
  PF_PID=""
}

# Round 1
echo ">>> Round 1: ${PORT_FORWARD_1} (${MODELS_1[*]})"
start_port_forward "$PORT_FORWARD_1" || exit 1
trap stop_port_forward EXIT
run_profiles "${MODELS_1[@]}"
stop_port_forward
trap - EXIT
echo ""

# Round 2
# echo ">>> Round 2: ${PORT_FORWARD_2} (${MODELS_2[*]})"
# start_port_forward "$PORT_FORWARD_2" || exit 1
# trap stop_port_forward EXIT
# run_profiles "${MODELS_2[@]}"
# stop_port_forward
# trap - EXIT
# echo ""

# Round 3
echo ">>> Round 3: ${PORT_FORWARD_3} (${MODELS_3[*]})"
start_port_forward "$PORT_FORWARD_3" || exit 1
trap stop_port_forward EXIT
run_profiles "${MODELS_3[@]}"
stop_port_forward
trap - EXIT

echo "Done. Results in profiling/result/"
