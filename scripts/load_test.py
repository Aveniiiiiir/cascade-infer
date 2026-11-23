"""HTTP 负载压测：QPS / TTFT / TPOT / P50-P95-P99（纯标准库，零依赖）。

用法:
    python scripts/load_test.py --url http://localhost:18923 \
        --concurrency 8 --requests 64 --max-tokens 64 --stream
    python scripts/load_test.py --url ... --api-key sk-test ...

流式模式测真实 TTFT（首 chunk 到达时间）与 TPOT（(总时-TTFT)/(token-1)）；
非流式只有一个总时延（TTFT 记为总时延）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request


def _one_request(url: str, api_key: str | None, prompt: str, max_tokens: int,
                 stream: bool) -> tuple[float, float, int, int]:
    """→ (ttft_s, total_s, n_tokens, ok)。"""
    body = json.dumps({
        "model": "cascade",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers=headers)
    t0 = time.perf_counter()
    ttft = -1.0
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            if not stream:
                data = json.loads(resp.read())
                total = time.perf_counter() - t0
                n = len(data["choices"][0]["message"]["content"].split())
                return total, total, n, 1
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0].get("delta", {}).get("content") or ""
                if delta and ttft < 0:
                    ttft = time.perf_counter() - t0
                n += max(1, len(delta.split())) if delta else 0
        total = time.perf_counter() - t0
        return (ttft if ttft >= 0 else total), total, n, 1
    except Exception:
        return 0.0, time.perf_counter() - t0, 0, 0


def run(args: argparse.Namespace) -> int:
    results: list[tuple[float, float, int, int]] = []
    lock = threading.Lock()
    t_start = time.perf_counter()

    def worker(k: int) -> None:
        for _ in range(args.requests // args.concurrency):
            r = _one_request(args.url, args.api_key, args.prompt, args.max_tokens, args.stream)
            with lock:
                results.append(r)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(args.concurrency)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t_start

    oks = [r for r in results if r[3]]
    errs = len(results) - len(oks)
    if not oks:
        print(f"ALL FAILED: {len(results)} requests, wall {wall:.1f}s")
        return 1

    ttfts = sorted(r[0] for r in oks)
    tpots = sorted((r[1] - r[0]) / max(1, r[2] - 1) for r in oks)
    toks = sum(r[2] for r in oks)

    def pct(xs: list[float], p: float) -> float:
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    print(f"requests: {len(results)}  ok: {len(oks)}  errors: {errs}")
    print(f"wall: {wall:.2f}s  qps: {len(oks) / wall:.2f}  tokens/s: {toks / wall:.1f}")
    print(f"ttft  p50={pct(ttfts, 0.50) * 1000:.0f}ms p95={pct(ttfts, 0.95) * 1000:.0f}ms "
          f"p99={pct(ttfts, 0.99) * 1000:.0f}ms mean={statistics.mean(ttfts) * 1000:.0f}ms")
    print(f"tpot  p50={pct(tpots, 0.50) * 1000:.1f}ms p95={pct(tpots, 0.95) * 1000:.1f}ms "
          f"mean={statistics.mean(tpots) * 1000:.1f}ms")
    return 0 if errs == 0 else 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Cascade HTTP load test (stdlib only)")
    ap.add_argument("--url", default="http://localhost:18923")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=64, help="总数（向下取整到并发倍数）")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--stream", action="store_true", help="SSE 流式（测真实 TTFT）")
    raise SystemExit(run(ap.parse_args()))


if __name__ == "__main__":
    main()