#!/usr/bin/env bash
# Cascade HTTP 服务启动脚本（detached + pid 文件 + 日志）。
#
# 用法（集群）:
#   CASCADE_MODEL=/root/autodl-tmp/DeepSeek-V2-Lite API_KEY=sk-xxx \
#     bash scripts/serve.sh            # 默认端口 8000
#   CASCADE_MODEL=... PORT=8080 bash scripts/serve.sh
#
# 停止: kill "$(cat ${PIDFILE:-/root/cascade_server.pid})"
# 日志: tail -f ${LOG:-/root/cascade_server.log}
#
# 说明:
# - 模型加载需 7-10 分钟，就绪前 /v1/models 会连接被拒（正常）。
# - 32GB 卡建议 MAX_LEN=4096 MAX_SEQS=32；48GB+ 可放宽。
set -euo pipefail

: "${CASCADE_MODEL:?需要设置 CASCADE_MODEL 指向本地 HF 格式模型目录}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"
MAX_SEQS="${MAX_SEQS:-32}"
API_KEY="${API_KEY:-}"
LOG="${LOG:-/root/cascade_server.log}"
PIDFILE="${PIDFILE:-/root/cascade_server.pid}"

args=(--model "$CASCADE_MODEL" --port "$PORT" --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS")
[ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")

setsid nohup python -m cascade.entrypoints.http.server "${args[@]}" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"

echo "started pid=$(cat "$PIDFILE") port=$PORT log=$LOG"
echo "等待就绪（模型加载 7-10 分钟）:"
echo "  until curl -sf http://localhost:$PORT/v1/models >/dev/null; do sleep 5; done"