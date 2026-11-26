#!/usr/bin/env bash
# 灰度压测：8/16/32 并发 SSE，记录 QPS / tokens/s / TTFT / TPOT。
# 用法: CASCADE_API_KEY=... bash scripts/load_test_suite.sh http://127.0.0.1:8080
set -euo pipefail

BASE="${1:-http://127.0.0.1:8080}"
: "${CASCADE_API_KEY:?需要设置 CASCADE_API_KEY}"
PYTHON_BIN="${PYTHON:-python3}"
[ -x /root/miniconda3/bin/python ] && PYTHON_BIN=/root/miniconda3/bin/python

for n in 8 16 32; do
  echo "===== concurrency=${n} ====="
  "$PYTHON_BIN" scripts/load_test.py \
    --url "$BASE" --api-key "$CASCADE_API_KEY" \
    --concurrency "$n" --requests 64 --max-tokens 32 --stream
done