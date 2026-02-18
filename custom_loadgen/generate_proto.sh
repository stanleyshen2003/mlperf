#!/usr/bin/env bash
# Regenerate gRPC Python stubs from proto/scheduler.proto
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p generated
uv run python -m grpc_tools.protoc \
  -I proto/ \
  --python_out=generated/ \
  --grpc_python_out=generated/ \
  proto/scheduler.proto
