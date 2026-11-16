"""e2e 分歧定位：同一 prompt 的 last-token logits 三方对拍。

Cascade explicit（HF oracle 路径） / Cascade absorbed（生产路径） / 官方 HF。
分阶段加载模型避免 32GB 显存同时驻留两份。
"""

from __future__ import annotations

import gc
import os
import sys

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")
PROMPT = os.environ.get("DEBUG_PROMPT", "The capital of France is")


def cascade_logits() -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    from cascade.config import CacheConfig
    from cascade.model_executor.model_loader import load_deepseek_v2
    from cascade.worker.cache_engine import CacheEngine

    model, cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = tok.encode(PROMPT)
    T = len(ids)
    cache = CacheEngine(
        cfg, CacheConfig(block_size=64), num_blocks=64, device="cuda", dtype=torch.bfloat16
    )
    positions = torch.arange(T, device="cuda")
    seq_ids = torch.zeros(T, dtype=torch.long, device="cuda")
    bt = torch.zeros(1, 8, dtype=torch.long, device="cuda")
    bt[0, 0] = 1
    x = torch.tensor(ids, device="cuda")
    with torch.no_grad():
        out_exp = model(x, positions, seq_ids, [T], bt, cache, use_absorbed=False)
        cache2 = CacheEngine(
            cfg, CacheConfig(block_size=64), num_blocks=64, device="cuda", dtype=torch.bfloat16
        )
        out_abs = model(x, positions, seq_ids, [T], bt, cache2, use_absorbed=True)
    exp = out_exp[-1].float().cpu()
    abs_ = out_abs[-1].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return exp, abs_, ids


def _hf_compat_patches() -> None:
    """官方 modeling_deepseek.py 在 transformers 4.5x 下的兼容补丁。"""
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


def hf_logits(ids: list[int]) -> torch.Tensor:
    _hf_compat_patches()
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].float()
    out = logits.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def report(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    diff = (a - b).abs()
    print(
        f"{name_a} vs {name_b}: argmax {a.argmax().item()} vs {b.argmax().item()}, "
        f"max|d|={diff.max().item():.4f}, mean|d|={diff.mean().item():.6f}, "
        f"top5 {a.topk(5).indices.tolist()} vs {b.topk(5).indices.tolist()}"
    )


def main() -> None:
    exp, abs_, ids = cascade_logits()
    print("cascade explicit vs absorbed:", end=" ")
    report("explicit", exp, "absorbed", abs_)
    hf = hf_logits(ids)
    report("cascade-explicit", exp, "hf", hf)
    report("cascade-absorbed", abs_, "hf", hf)


if __name__ == "__main__":
    main()
