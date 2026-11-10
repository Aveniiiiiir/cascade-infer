#!/usr/bin/env bash
# 集群冒烟：安装 + CPU 全量 + model 层正确性（不跑 e2e/benchmark）
# 用法: bash scripts/gpu_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -e ".[dev,so]" -q
python -m pytest tests/cpu tests/model -q

echo "SMOKE OK"
