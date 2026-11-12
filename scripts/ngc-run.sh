#!/usr/bin/env bash
#
# Run a command inside the NGC PyTorch container with the GB10 GPU attached.
#
#   scripts/ngc-run.sh python -m cv.simplecnn.train
#   scripts/ngc-run.sh bash
#
# Environment:
#   DLF_IMAGE  image tag (default: dlf-ngc:26.09)
#   HF_CACHE   host HuggingFace cache directory (default: ~/.cache/huggingface)
#
# Build the image once with:
#   docker build -t dlf-ngc:26.09 -f docker/ngc-pytorch.Dockerfile docker/
set -euo pipefail

usage() {
  cat >&2 <<'TXT'
usage: scripts/ngc-run.sh <command> [args...]

Runs <command> in the dlf-ngc:26.09 container (NGC PyTorch + scikit-learn) with
the GPU attached and this repository mounted at /workspace.

examples:
  scripts/ngc-run.sh python -m cv.simplecnn.train
  scripts/ngc-run.sh bash
TXT
}

if [ "$#" -eq 0 ]; then
  usage
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${DLF_IMAGE:-dlf-ngc:26.09}
hf_cache=${HF_CACHE:-$HOME/.cache/huggingface}

if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "error: image '$image' not found; build it first:" >&2
  echo "  docker build -t $image -f docker/ngc-pytorch.Dockerfile docker/" >&2
  exit 1
fi

mkdir -p "$hf_cache"

# -it only on a real TTY, so non-interactive runs (agents, CI) never wait on stdin.
tty=()
if [ -t 0 ] && [ -t 1 ]; then
  tty=(-it)
fi

exec docker run --rm "${tty[@]}" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$repo_root:/workspace" \
  -w /workspace \
  -v "$hf_cache:/hf-cache" \
  -e HF_HOME=/hf-cache \
  -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 \
  -u "$(id -u):$(id -g)" \
  "$image" "$@"
