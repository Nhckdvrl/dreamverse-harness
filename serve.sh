#!/usr/bin/env bash
# Usage: serve.sh <gpu-index> [port]. Runs the OmniRef server in the foreground on this node.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="$1"
PORT="${2:-8000}"
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN_H3
export FASTVIDEO_H3_VSA_FP4=1 FASTVIDEO_H3_VSA_FP4_PREFIX=fp8
export FASTVIDEO_H3_VSA_FP4_TWO_LEVEL_P=1 FASTVIDEO_H3_VSA_FP4_K_SMOOTH=1
export FASTVIDEO_H3_GATE_CHUNK_ROWS=8192
export FASTVIDEO_MINIMAX_H3_FUSIONS=all
export FASTVIDEO_VSA_TRITON=1 FASTVIDEO_VSA_PTX_BLACKWELL=0 FASTVIDEO_FA4=0
export FASTVIDEO_H3_VAE_TILE_BATCH=8
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$HOME/projects/FastVideo-dv/.venv/bin/activate"
exec fastvideo serve --config serve.yaml --server.port "$PORT"
