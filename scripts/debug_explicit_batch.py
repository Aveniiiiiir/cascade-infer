"""explicit 路径在真实 20 段 prefill 批下的逐层定位。"""

from __future__ import annotations

import os

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")
PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "def fibonacci(n):",
    "中国的首都是",
    "Explain the difference between TCP and UDP in one paragraph:",
    "1 + 1 =",
    "Q: What is the largest planet? A:",
    "import numpy as np\nnp.array([1,2,3]).",
    "The theory of relativity states that",
    "Translate to English: 你好世界",
    "SELECT * FROM users WHERE",
    "In a shocking finding, scientists discovered a herd of unicorns living",
    "中国的四大发明包括",
    "for i in range(10):",
    "The quick brown fox",
    "Water boils at",
    "Machine learning is",
    "The Great Wall of China was built",
    "print(",
    "A B C D E F G",
]


def main() -> None:
    from transformers import AutoTokenizer

    from cascade.config import CacheConfig
    from cascade.model_executor.model_loader import load_deepseek_v2
    from cascade.worker.cache_engine import CacheEngine

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model, cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    segs = [tok.encode(p) for p in PROMPTS]
    lens = [len(s) for s in segs]
    print("segment lens:", lens, "total", sum(lens))
    flat = torch.tensor([t for s in segs for t in s], device="cuda")
    positions = torch.cat([torch.arange(n, device="cuda") for n in lens])
    seq_ids = torch.cat(
        [torch.full((n,), i, dtype=torch.long, device="cuda") for i, n in enumerate(lens)]
    )
    bt = torch.zeros(len(segs), 8, dtype=torch.long, device="cuda")
    for i in range(len(segs)):
        bt[i, 0] = i + 1  # 每段 1 块（64 容量）
    cache = CacheEngine(
        cfg, CacheConfig(block_size=64), num_blocks=64, device="cuda", dtype=torch.bfloat16
    )
    x = model.model.embed_tokens(flat)
    for i, layer in enumerate(model.model.layers):
        try:
            x = layer(x, positions, seq_ids, lens, bt, cache, model.model.rope, False, None, False)
            torch.cuda.synchronize()
            print(f"layer {i} OK", flush=True)
        except Exception as e:
            print(f"layer {i} FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
            raise


if __name__ == "__main__":
    main()
