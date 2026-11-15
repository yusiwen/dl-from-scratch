#!/usr/bin/env bash
#
# Open the whole repository in JupyterLab, inside the GPU container.
#
#   scripts/jupyter.sh                  # http://127.0.0.1:8888/lab?token=<generated>
#   scripts/jupyter.sh --port 9000
#
# The server runs in the same image the notebooks are tested in (GPU, gymnasium,
# the repository mounted at /workspace, the Hugging Face cache mounted at
# /hf-cache), so a notebook behaves here exactly as it does under
# scripts/run_notebooks.sh -- just interactively.
#
# Environment:
#   DLF_IMAGE      image tag (default: dlf-nb:26.09)
#   HF_CACHE       host Hugging Face cache (default: /tmp/hf-ngc)
#   JUPYTER_TOKEN  fixed token instead of a random one
set -euo pipefail

port=8888
host=127.0.0.1

usage() {
  cat >&2 <<'TXT'
usage: scripts/jupyter.sh [--port <host-port>]

Starts JupyterLab in the notebook container with this repository as the root
directory, and prints the URL (with token) to open.

examples:
  scripts/jupyter.sh
  scripts/jupyter.sh --port 9000
TXT
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) port=${2:?--port needs a value}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage; exit 2 ;;
  esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image=${DLF_IMAGE:-dlf-nb:26.09}
hf_cache=${HF_CACHE:-/tmp/hf-ngc}

if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "error: image '$image' not found; build it first:" >&2
  echo "  printf 'FROM dlf-ngc:26.09\nRUN pip install --no-cache-dir gymnasium\n'" >&2
  echo "    | docker build -t dlf-nb:26.09 -f - ." >&2
  exit 1
fi

# A token is generated per start: the port is published on the loopback
# interface only, but a token keeps the kernel reachable from no one else.
token=${JUPYTER_TOKEN:-$(openssl rand -hex 16)}
mkdir -p "$hf_cache"

cat <<TXT
JupyterLab: http://$host:$port/lab?token=$token

  image      $image
  repository $repo_root  ->  /workspace
  HF cache   $hf_cache  ->  /hf-cache
  kernel     python3 (the container's interpreter, GPU visible)

Stop it with Ctrl-C.
TXT

exec docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "$host:$port:8888" \
  -v "$repo_root:/workspace" \
  -w /workspace \
  -v "$hf_cache:/hf-cache" \
  -e HF_HOME=/hf-cache \
  -e MPLBACKEND=Agg \
  -e MPLCONFIGDIR=/tmp/mpl \
  -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 \
  -u "$(id -u):$(id -g)" \
  "$image" jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8888 \
    --ServerApp.root_dir=/workspace \
    --ServerApp.allow_remote_access=True \
    --ServerApp.password='' \
    --IdentityProvider.token="$token"
