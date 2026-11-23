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

CASCADE_CUDAGRAPH=0 python -m cascade.entrypoints.llm --model "$CASCADE_E2E_MODEL" \
    --prompt "The capital of France is" --prompt "def hello():" --max-tokens 16

# HTTP serving 冒烟（后台起服务 → 就绪探测 → 1 条请求 → 关闭）
# 2026-11-18 修复：原 `sleep 30` 对真实模型必然失败——V2-Lite 从数据盘
# 加载需 ~400s；且 set -e 失败路径不清理服务进程。改为就绪探测（最长
# 600s）+ trap 兜底清理。
CASCADE_CUDAGRAPH=0 python -m cascade.entrypoints.http.server --model "$CASCADE_E2E_MODEL" --port 18923 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
trap cleanup EXIT
ready=0
for _ in $(seq 1 450); do
  if curl -s --max-time 2 http://localhost:18923/v1/models | grep -q model; then
    ready=1; break
  fi
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "HTTP server died during startup"; exit 1
  fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo "HTTP server not ready within 900s"; exit 1
fi
curl -s -X POST http://localhost:18923/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' | grep -q chat.completion
trap - EXIT
cleanup

echo "ACCEPTANCE FULL OK"
