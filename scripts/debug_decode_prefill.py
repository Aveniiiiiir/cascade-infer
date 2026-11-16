"""decode vs re-prefill、混合批 vs 单批 的 logits 对拍。"""

from __future__ import annotations

import os

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")
PROMPTS = ["The capital of France is", "Once upon a time,"]


def make_cache(cfg, blocks: int = 64):
    from cascade.config import CacheConfig
    from cascade.worker.cache_engine import CacheEngine

    return CacheEngine(cfg, CacheConfig(block_size=64), num_blocks=blocks, device="cuda", dtype=torch.bfloat16)


def run(model, cfg, segs: list[list[int]], start_positions: list[int], blocks_per_seq: int = 8):
    """segs: 每段 token ids；start_positions: 每段起始绝对位置。返回每段末行 logits。"""
    cache = make_cache(cfg)
    lens = [len(s) for s in segs]
    flat = torch.tensor([t for s in segs for t in s], device="cuda")
    positions = torch.cat(
        [torch.arange(sp, sp + n, device="cuda") for sp, n in zip(start_positions, lens)]
    )
    seq_ids = torch.cat(
        [torch.full((n,), i, dtype=torch.long, device="cuda") for i, n in enumerate(lens)]
    )
    bt = torch.zeros(len(segs), 16, dtype=torch.long, device="cuda")
    blk = 1
    for i in range(len(segs)):
        n_need = (start_positions[i] + lens[i] + 63) // 64
        bt[i, :n_need] = torch.arange(blk, blk + n_need, device="cuda")
        blk += n_need
    off = 0
    ends = []
    for n in lens:
        off += n
        ends.append(off - 1)
    with torch.no_grad():
        logits = model(flat, positions, seq_ids, lens, bt, cache, use_absorbed=True)
    return [logits[e].float() for e in ends]


def main() -> None:
    from transformers import AutoTokenizer

    from cascade.model_executor.model_loader import load_deepseek_v2

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model, cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()

    ids = tok.encode(PROMPTS[0])
    la = run(model, cfg, [ids], [0])[0]
    a0 = int(la.argmax())
    print("prefill argmax:", a0)

    # 1) decode 一步 vs 全量 re-prefill
    lb = run(model, cfg, [[a0]], [len(ids)])[0]
    lc = run(model, cfg, [ids + [a0]], [0])[0]
    print(
        "decode vs re-prefill: argmax", int(lb.argmax()), "vs", int(lc.argmax()),
        "max|d|", float((lb - lc).abs().max()),
        "top5", lb.topk(5).indices.tolist(), "vs", lc.topk(5).indices.tolist(),
    )

    # 2) 混合批 vs 单独
    ids2 = tok.encode(PROMPTS[1])
    mixed = run(model, cfg, [ids, ids2], [0, 0])
    solo1 = run(model, cfg, [ids], [0])[0]
    solo2 = run(model, cfg, [ids2], [0])[0]
    for name, m, s in (("seq0", mixed[0], solo1), ("seq1", mixed[1], solo2)):
        print(
            f"mixed vs solo {name}: argmax {int(m.argmax())} vs {int(s.argmax())}, "
            f"max|d|={float((m - s).abs().max()):.4f}, "
            f"top5 {m.topk(5).indices.tolist()} vs {s.topk(5).indices.tolist()}"
        )


if __name__ == "__main__":
    main()
