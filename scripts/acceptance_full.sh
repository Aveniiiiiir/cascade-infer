#!/usr/bin/env bash
# 全部 Phase 的集群验收（P1-P6）：
#   1. CPU 调度全量（Windows 相同守卫）
#   2. model/triton_ops/engine 层正确性（kernel vs ref、ZOS、CUDA graph、
#      MTP 贪心等价、SO、EPLB、HTTP server）
#   3. e2e：V2-Lite 贪心 vs HF 逐 token + 批不变 + 同 seed 复现
#   4. 离线 CLI 冒烟 + HTTP serving 冒烟
# 用法: CASCADE_E2E_MODEL=/data/DeepSeek-V2-Lite bash scripts/acceptance_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CASCADE_E2E_MODEL:?需要设置 CASCADE_E2E_MODEL（HF 格式 V2-Lite 目录）}"

pip install -e ".[dev,so,serve]" -q
python -m pytest tests/cpu -q
python -m pytest tests/model tests/triton_ops tests/engine -q
python -m pytest tests/e2e -q

python -m cascade.entrypoints.llm --model "$CASCADE_E2E_MODEL" \
    --prompt "The capital of France is" --prompt "def hello():" --max-tokens 16

# HTTP serving 冒烟（后台起服务 → 1 条请求 → 关闭）
python -m cascade.entrypoints.http.server --model "$CASCADE_E2E_MODEL" --port 18923 &
SRV=$!
sleep 30
curl -s http://localhost:18923/v1/models | grep -q model
curl -s -X POST http://localhost:18923/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' | grep -q chat.completion
kill $SRV

echo "ACCEPTANCE FULL OK"
