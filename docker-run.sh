#!/usr/bin/env bash
# Usage: ./docker-run.sh sweep-vllm sweep-others.json
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p realtime_runs_vllm_0709 "$HOME/.cache/huggingface" "$HOME/.cache/flashinfer-docker"

docker run --rm -it --gpus all --ipc=host \
  -v "$PWD/.env:/app/.env:ro" \
  -v "$PWD/video.mp4:/app/video.mp4:ro" \
  -v "$PWD/sweep-others.json:/app/sweep-others.json:ro" \
  -v "$PWD/sweep-single.json:/app/sweep-single.json:ro" \
  -v "$PWD/realtime_runs_vllm_0709:/app/realtime_runs_vllm_0709" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/flashinfer-docker:/root/.cache/flashinfer" \
  realtime-bench "$@"
