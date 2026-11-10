#!/usr/bin/env bash
# Phase 1 + 2 验收（集群）：
#   1. CPU 调度全量（与 Windows 相同守卫）
#   2. model 层不变式（显式≡吸收、增量≡整段、chunked、批不变、MoE oracle）
#   3. e2e：V2-Lite 贪心 vs HF 逐 token 一致（需 CASCADE_E2E_MODEL）
#   4. 离线生成 CLI 冒烟
# 用法: CASCADE_E2E_MODEL=/data/DeepSeek-V2-Lite bash scripts/acceptance_p2.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CASCADE_E2E_MODEL:?需要设置 CASCADE_E2E_MODEL（HF 格式 V2-Lite 目录）}"

pip install -e ".[dev,so]" -q
python -m pytest tests/cpu -q
python -m pytest tests/model -q
python -m pytest tests/e2e -q

# 离线 CLI 冒烟
python -m cascade.entrypoints.llm --model "$CASCADE_E2E_MODEL" \
    --prompt "The capital of France is" --prompt "def hello():" --max-tokens 16

echo "ACCEPTANCE P1+P2 OK"
