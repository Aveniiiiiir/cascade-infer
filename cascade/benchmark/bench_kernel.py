"""逐 kernel 微基准（GB/s / 时延）——性能回归基线工具（集群 GPU）。

用法:
    python -m benchmark.bench_kernel            # 全部
    python -m benchmark.bench_kernel decode moe # 指定算子
"""

from __future__ import annotations

import sys
import time

import torch

from cascade.model_executor.triton_ops.det_random_sampling import det_random_sampling
from cascade.model_executor.triton_ops.fused_moe import fused_experts
from cascade.model_executor.triton_ops.paged_mla_decode_attn import paged_mla_decode_attn
from cascade.model_executor.triton_ops.paged_mla_prefill_attn import paged_mla_prefill_attn
from cascade.model_executor.triton_ops.store_paged_kv import store_paged_kv

# V2-Lite 形状基线
H, KV_LORA, ROPE, D = 16, 512, 64, 576
E, N, HID = 64, 1408, 2048
SCALE = (128 + 64) ** -0.5


def bench(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench_store_kv() -> dict:
    bs, num_blocks, T = 64, 4096, 2048
    kv = torch.zeros(num_blocks, bs, D, dtype=torch.bfloat16, device="cuda")
    latent = torch.randn(T, D, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, 32, (T,), dtype=torch.int64, device="cuda")
    pos = torch.randint(0, 4000, (T,), dtype=torch.int64, device="cuda")
    bt = torch.randint(1, num_blocks, (32, 64), dtype=torch.int64, device="cuda")
    ms = bench(lambda: store_paged_kv(kv, latent, ids, pos, bt))
    gbs = T * D * 2 / ms / 1e6
    return {"op": "store_paged_kv", "ms": ms, "gbps": gbs}


def bench_decode_attn() -> dict:
    n_seqs, seqlen, bs = 256, 4096, 64
    # 池大小必须覆盖 n_seqs × seqlen/bs 个块（此前写死 4096，块 id 越界 →
    # illegal memory access；审查补充：块表测试数据需与池容量自洽）
    num_blocks = n_seqs * (seqlen // bs) + 1
    q = torch.randn(n_seqs, H, D, dtype=torch.bfloat16, device="cuda") * 0.05
    kv = torch.randn(num_blocks, bs, D, dtype=torch.bfloat16, device="cuda") * 0.05
    bt = torch.zeros(n_seqs, seqlen // bs, dtype=torch.int64, device="cuda")
    for i in range(n_seqs):
        bt[i] = torch.arange(i * (seqlen // bs) + 1, (i + 1) * (seqlen // bs) + 1, device="cuda")
    sl = torch.full((n_seqs,), seqlen, dtype=torch.int64, device="cuda")
    ms = bench(lambda: paged_mla_decode_attn(q, kv, bt, sl, SCALE, KV_LORA))
    # KV 加载为带宽下界：n_seqs × seqlen × D × 2B
    gbs = n_seqs * seqlen * D * 2 / ms / 1e6
    return {"op": "paged_mla_decode_attn", "ms": ms, "kv_gbps": gbs, "tokens_per_s": n_seqs * 1 / (ms / 1e3)}


def bench_prefill_attn() -> dict:
    T, seqlen, bs = 2048, 2048, 64
    num_blocks = 4096
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda") * 0.05
    kv = torch.randn(num_blocks, bs, D, dtype=torch.bfloat16, device="cuda") * 0.05
    bt = torch.zeros(8, seqlen // bs, dtype=torch.int64, device="cuda")
    for i in range(8):
        bt[i] = torch.arange(i * 32 + 1, i * 32 + 33, device="cuda")
    sl = torch.full((8,), seqlen, dtype=torch.int64, device="cuda")
    segments = [(i, 0, T // 8) for i in range(8)]
    ms = bench(lambda: paged_mla_prefill_attn(q, kv, bt, sl, segments, SCALE, KV_LORA))
    return {"op": "paged_mla_prefill_attn", "ms": ms, "prefill_tok_per_s": T / (ms / 1e3)}


def bench_moe() -> dict:
    T, K = 512, 6
    x = torch.randn(T, HID, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, N, HID, dtype=torch.bfloat16, device="cuda") * 0.02
    w3 = torch.randn(E, N, HID, dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn(E, HID, N, dtype=torch.bfloat16, device="cuda") * 0.02
    ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
    wts = torch.rand(T, K, dtype=torch.float32, device="cuda")
    ms = bench(lambda: fused_experts(x, w1, w3, w2, wts, ids))
    flops = T * K * 3 * 2 * N * HID  # 3 个 GEMM × 2×MNK
    return {"op": "fused_moe", "ms": ms, "tflops": flops / ms / 1e9}


def bench_sampling() -> dict:
    T, vocab = 256, 129280
    probs = torch.softmax(torch.randn(T, vocab, device="cuda"), dim=-1)
    seeds = torch.arange(T, dtype=torch.int64, device="cuda")
    pos = torch.arange(T, dtype=torch.int64, device="cuda")
    ms = bench(lambda: det_random_sampling(probs, seeds, pos))
    return {"op": "det_random_sampling", "ms": ms}


BENCHES = {
    "store": bench_store_kv,
    "decode": bench_decode_attn,
    "prefill": bench_prefill_attn,
    "moe": bench_moe,
    "sample": bench_sampling,
}


def main() -> None:
    torch.manual_seed(0)
    names = sys.argv[1:] or list(BENCHES)
    print(f"{'op':28s} {'ms':>10s}  metric")
    for name in names:
        r = BENCHES[name]()
        metric = {k: f"{v:.3g}" for k, v in r.items() if k not in ("op", "ms")}
        extra = " ".join(f"{k}={v}" for k, v in metric.items())
        print(f"{r['op']:28s} {r['ms']:>10.3f}  {extra}")


if __name__ == "__main__":
    main()
