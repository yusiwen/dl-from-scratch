#!/usr/bin/env bash
#
# Execute notebooks headlessly with the GB10 GPU attached.
#
#   scripts/run_notebooks.sh                                  # every notebook
#   scripts/run_notebooks.sh --only ml/basics/ --timeout 300   # a subset
#
# Build the image once (adds gymnasium on top of the NGC base):
#   printf 'FROM dlf-ngc:26.09\nRUN pip install --no-cache-dir gymnasium\n' \
#     | docker build -t dlf-nb:26.09 -f - .
#
# Environment:
#   DLF_IMAGE  image tag (default: dlf-nb:26.09)
#   HF_CACHE   host dataset cache (default: /tmp/hf-ngc)
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${DLF_IMAGE:-dlf-nb:26.09}
hf_cache=${HF_CACHE:-/tmp/hf-ngc}
mkdir -p "$hf_cache"

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
  -e MPLBACKEND=Agg \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 \
  -u "$(id -u):$(id -g)" \
  "$image" python scripts/run_notebooks.py "$@"
