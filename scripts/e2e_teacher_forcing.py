"""teacher-forcing 对拍：喂 HF 的完整序列，逐位置比较 argmax。

free-running 的 6-7/20 分歧可能是「单步数值噪声在 32 步累积后分叉」。
本脚本消除累积：两侧都在 HF 的 token 序列上前向，逐位置比较
argmax（即「下一步预测」），并统计分歧位置的 logit margin。
"""

from __future__ import annotations

import gc
import os
import sys

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")
MAX_TOKENS = 32

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


def _hf_patches() -> None:
    import importlib.machinery
    import types

    if "flash_attn" not in sys.modules:
        fa = types.ModuleType("flash_attn")
        fa.__version__ = "0.0.0"
        fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
        fa.flash_attn_func = lambda *a, **k: None
        fa.flash_attn_varlen_func = lambda *a, **k: None
        sys.modules["flash_attn"] = fa
    import transformers.utils.import_utils as iu

    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda: False
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, seq_length: self.get_seq_length()


def hf_generate_and_logits(ids_list: list[list[int]]):
    _hf_patches()
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    outs = []
    with torch.no_grad():
        for ids in ids_list:
            seq = list(ids)
            for _ in range(MAX_TOKENS):
                x = torch.tensor([seq], device="cuda")
                nxt = int(model(x, use_cache=False).logits[0, -1].argmax())
                seq.append(nxt)
                if nxt == 1:  # eos 近似（真实 eos=100001，这里不早停）
                    break
            logits = model(torch.tensor([seq], device="cuda"), use_cache=False).logits[0].float().cpu()
            outs.append((seq, logits))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outs


def cascade_logits(seqs: list[list[int]]):
    from cascade.config import CacheConfig
    from cascade.model_executor.model_loader import load_deepseek_v2
    from cascade.worker.cache_engine import CacheEngine

    model, cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    outs = []
    with torch.no_grad():
        for seq in seqs:
            T = len(seq)
            cache = CacheEngine(
                cfg, CacheConfig(block_size=64), num_blocks=64, device="cuda", dtype=torch.bfloat16
            )
            bt = torch.zeros(1, 4, dtype=torch.long, device="cuda")
            bt[0, 0] = 1
            logits = model(
                torch.tensor(seq, device="cuda"),
                torch.arange(T, device="cuda"),
                torch.zeros(T, dtype=torch.long, device="cuda"),
                [T], bt, cache, use_absorbed=False,
            ).float().cpu()  # [T, V]（模型返回展平行，无 batch 维）
            outs.append(logits)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outs


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids_list = [tok.encode(p) for p in PROMPTS]
    hf = hf_generate_and_logits(ids_list)
    seqs = [s for s, _ in hf]
    casc = cascade_logits(seqs)

    total = match = 0
    worst = []
    for i, ((seq, hlog), clog) in enumerate(zip(hf, casc)):
        T = len(seq)
        m = 0
        n = 0
        bad = []
        for p in range(len(ids_list[i]) - 1, T - 1):
            a = int(clog[p].argmax())
            b = int(hlog[p].argmax())
            n += 1
            if a == b:
                m += 1
            else:
                top2 = hlog[p].topk(2).values
                bad.append((p, a, b, float(top2[0] - top2[1])))
        total += n
        match += m
        if bad:
            worst.append((PROMPTS[i], m, n, bad[:3]))
        print(f"[{i:2d}] match {m}/{n} " + (f"bad={bad[:2]}" if bad else ""), flush=True)

    print(f"OVERALL per-step argmax match: {match}/{total} = {match / total:.4f}")
    print("worst prompts:", worst[:5])


if __name__ == "__main__":
    main()
