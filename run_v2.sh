#!/usr/bin/env bash
# Render runs/v2 on the two fvcrc21 servers: P1 on :8121, P2 on :8122. sax is split by seed; P2 waits for P1's cuts.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
run() { uv run -q --with opencv-python-headless --with numpy --with scikit-image python run_eval.py --out runs/v2 "$@"; }
p1() { run --server http://127.0.0.1:8121 --scenes skate && run --server http://127.0.0.1:8121 --scenes sax --seeds 1234; }
p2() { run --server http://127.0.0.1:8122 --scenes rider || return 1
       until [ -d runs/v2/sax/last_frame ]; do sleep 10; done
       run --server http://127.0.0.1:8122 --scenes sax --seeds 42; }
(p1 > logs/run-v2-p1.log 2>&1; echo "exit=$?" >> logs/run-v2-p1.log) &
(p2 > logs/run-v2-p2.log 2>&1; echo "exit=$?" >> logs/run-v2-p2.log) &
wait
