#!/usr/bin/env bash
# Usage: serve.sh <stable-audio|mmaudio> <gpu-index> <port>. Runs one FastVideo audio server in the foreground.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="$2"
source "$HOME/projects/FastVideo-dv/.venv/bin/activate"
exec fastvideo serve --config "serve-$1.yaml" --server.port "$3"
