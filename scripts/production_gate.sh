#!/usr/bin/env bash
# 生产灰度门禁：鉴权 / 推理 / 指标 / 反代健康检查（无第三方依赖）。
# 用法: CASCADE_API_KEY=... bash scripts/production_gate.sh http://127.0.0.1:8080
set -euo pipefail

BASE="${1:-http://127.0.0.1:8080}"
: "${CASCADE_API_KEY:?需要设置 CASCADE_API_KEY（与分析实例上的 key 一致）}"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; exit 1; }

code="$(curl -s -o "$TMP" -w '%{http_code}' --max-time 5 "$BASE/v1/models" || true)"
[ "$code" = "401" ] && pass "无鉴权返回 401" || fail "无鉴权期望 401，实际 $code"

code="$(curl -s -o "$TMP" -w '%{http_code}' --max-time 5 -H "Authorization: Bearer $CASCADE_API_KEY" "$BASE/v1/models" || true)"
[ "$code" = "200" ] && grep -q '"object"' "$TMP" && pass "鉴权 /v1/models 200" || fail "鉴权 /v1/models $code"

resp="$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $CASCADE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"cascade","messages":[{"role":"user","content":"用一句话解释 MLA"}],"max_tokens":16,"temperature":0.0}' || true)"
printf '%s' "$resp" | grep -q 'chat.completion' && pass "chat/completions 推理成功" || fail "chat/completions 失败: ${resp:0:200}"

metrics="$(curl -s --max-time 5 -H "Authorization: Bearer $CASCADE_API_KEY" "$BASE/metrics" || true)"
printf '%s' "$metrics" | grep -q '^cascade_requests_total' && pass "Prometheus 指标导出正常" || fail "/metrics 缺少 cascade_requests_total"

if hcode="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 'http://127.0.0.1:8080/healthz' || true)"; then
  case "$hcode" in
    200|401) pass "反代 /healthz 可达（HTTP $hcode，后端鉴权正常）" ;;
    *) fail "反代 /healthz 异常（HTTP $hcode）" ;;
  esac
fi

echo "GATE OK"