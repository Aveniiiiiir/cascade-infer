#!/usr/bin/env bash
# 容器内实际部署入口：supervisord 守护 Cascade + nginx 内网反代。
# 依赖 /etc/cascade-infer/{env,supervisord.conf,nginx.conf,cascade.conf} 已安装。
set -euo pipefail
set -a
# shellcheck disable=SC1091
. /etc/cascade-infer/env
set +a

mkdir -p /var/log/cascade-infer

SUPERVISORCTL=/root/miniconda3/bin/supervisorctl
SUPERVISORD=/root/miniconda3/bin/supervisord
SUPERVISOR_CONF=/etc/cascade-infer/supervisord.conf

if ! "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" status >/dev/null 2>&1; then
  "$SUPERVISORD" -c "$SUPERVISOR_CONF"
fi

echo "waiting for Cascade /v1/models (log: /var/log/cascade-infer/service.log)"
ready=0
for _ in $(seq 1 600); do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
    -H "Authorization: Bearer $CASCADE_API_KEY" http://127.0.0.1:8000/v1/models || true)"
  if [ "$code" = "200" ]; then ready=1; break; fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo "Cascade not ready within 1200s"
  "$SUPERVISORCTL" -c "$SUPERVISOR_CONF" status || true
  tail -40 /var/log/cascade-infer/service.log || true
  exit 1
fi
echo "Cascade ready"

NGINX=/opt/nginx/sbin/nginx
NGINX_CONF=/etc/cascade-infer/nginx.conf
"$NGINX" -t -c "$NGINX_CONF"
if [ -f /run/cascade-nginx.pid ] && kill -0 "$(cat /run/cascade-nginx.pid)" 2>/dev/null; then
  "$NGINX" -s reload -c "$NGINX_CONF"
else
  "$NGINX" -c "$NGINX_CONF"
fi
sleep 1
echo "nginx started on :8080"
curl -s -o /dev/null -w 'proxy /v1/models -> HTTP %{http_code}\n' \
  -H "Authorization: Bearer $CASCADE_API_KEY" http://127.0.0.1:8080/v1/models