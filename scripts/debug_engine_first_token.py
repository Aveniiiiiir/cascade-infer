"""引擎路径最小复现：单 prompt generate 的首 token vs 直接 forward 的 argmax。"""

from __future__ import annotations

import os

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")
PROMPT = os.environ.get("DEBUG_PROMPT", "The capital of France is")


def main() -> None:
    from cascade.config import CacheConfig
    from cascade.entrypoints.llm import LLM
    from cascade.worker.cache_engine import CacheEngine

    llm = LLM(MODEL_DIR, max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=4096)
    ids = llm.tokenizer.encode(PROMPT)
    print("prompt ids:", ids)

    # --- 引擎路径 ---
    out = llm.generate([PROMPT], max_tokens=6, temperature=0.0)
    _gen_ids = llm.tokenizer.encode(out[0]) if out[0] else []
    print("engine generate text:", repr(out[0]))

    # 用实际 token ids 重新跑一次拿 ids（generate 返回文本，直接取引擎输出）
    from cascade.sampling_params import SamplingParams

    eng = llm._engine
    rid = eng.add_request(ids, SamplingParams(temperature=0.0, max_tokens=8, seq_seed=0))
    res: list[int] = []
    while eng.scheduler.has_unfinished_requests:
        for ro in eng.step():
            if ro.request_id == rid:
                res.extend(ro.new_token_ids)
    print("engine greedy token ids:", res[:8])

    # --- 直接 forward 对照（同模型实例）---
    from cascade.config import ModelConfig  # noqa: F401

    model = llm._engine.runner.model
    cfg = llm._engine.runner.cfg
    T = len(ids)
    cache = CacheEngine(
        cfg, CacheConfig(block_size=64), num_blocks=64, device="cuda", dtype=torch.bfloat16
    )
    positions = torch.arange(T, device="cuda")
    seq_ids = torch.zeros(T, dtype=torch.long, device="cuda")
    bt = torch.zeros(1, 8, dtype=torch.long, device="cuda")
    bt[0, 0] = 1
    with torch.no_grad():
        logits = model(
            torch.tensor(ids, device="cuda"), positions, seq_ids, [T], bt, cache,
            use_absorbed=True,
        )
    print("direct forward argmax:", int(logits[-1].float().argmax()))
    top5 = logits[-1].float().topk(5).indices.tolist()
    print("direct forward top5:", top5)


if __name__ == "__main__":
    main()
