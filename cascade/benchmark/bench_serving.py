"""serving 基准（stdlib 线程 + urllib；集群对 /v1/chat/completions 压测）。

用法:
    python -m benchmark.bench_serving --url http://localhost:8000 \
        --concurrency 16 --num-prompts 64 --max-tokens 128
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request


def _one_request(url: str, prompt: str, max_tokens: int, results: list, idx: int) -> None:
    body = json.dumps(
        {
            "model": "bench",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        n_tok = data.get("usage", {}).get("completion_tokens", 0)
        results[idx] = {"ok": True, "latency_s": time.perf_counter() - t0, "tokens": n_tok}
    except Exception as e:
        results[idx] = {"ok": False, "latency_s": time.perf_counter() - t0, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Write a short story about a robot.")
    args = parser.parse_args()

    results: list = [None] * args.num_prompts
    threads: list[threading.Thread] = []
    t0 = time.perf_counter()
    for i in range(args.num_prompts):
        th = threading.Thread(target=_one_request, args=(args.url, args.prompt, args.max_tokens, results, i))
        # 控制并发窗
        while sum(t.is_alive() for t in threads) >= args.concurrency:
            time.sleep(0.01)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r and r["ok"]]
    lat = sorted(r["latency_s"] for r in ok)
    toks = sum(r["tokens"] for r in ok)
    print(f"requests: {len(ok)}/{args.num_prompts} ok, wall {wall:.1f}s")
    if lat:
        print(f"latency p50 {lat[len(lat)//2]:.2f}s p99 {lat[int(len(lat)*0.99)]:.2f}s")
    print(f"output throughput: {toks / wall:.1f} tok/s")
    print(f"request throughput: {len(ok) / wall:.1f} req/s")


if __name__ == "__main__":
    main()
