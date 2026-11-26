#!/usr/bin/env bash
# 由 supervisord/systemd 调用的服务入口：读取 root-only env 后 exec Python。
set -euo pipefail
set -a
# shellcheck disable=SC1091
. /etc/cascade-infer/env
set +a
cd /root/cascade-release-327b3b4
exec /root/miniconda3/bin/python -m cascade.entrypoints.http.server \
  --model "$CASCADE_MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len "${MAX_LEN:-4096}" \
  --max-num-seqs "${MAX_SEQS:-24}" \
  --api-key "$CASCADE_API_KEY"