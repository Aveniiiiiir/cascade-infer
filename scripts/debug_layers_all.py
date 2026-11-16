"""逐层输出对拍：Cascade（explicit）vs 官方 HF，定位首个显著分歧层。"""

from __future__ import annotations

import gc
import os
import sys

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")
PROMPT = "The capital of France is"


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


def hf_layers(ids: list[int]) -> list[torch.Tensor]:
    _hf_patches()
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    outs: list[torch.Tensor] = []
    hs = []
    for layer in model.model.layers:
        hs.append(layer.register_forward_hook(
            lambda _m, _i, o: outs.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu())
        ))
    with torch.no_grad():
        model(torch.tensor([ids], device="cuda"), use_cache=False)
    for h in hs:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outs


def cascade_layers(ids: list[int]) -> list[torch.Tensor]:
    from cascade.config import CacheConfig
    from cascade.model_executor.model_loader import load_deepseek_v2
    from cascade.worker.cache_engine import CacheEngine

    model, cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    T = len(ids)
    x = model.model.embed_tokens(torch.tensor(ids, device="cuda"))
    positions = torch.arange(T, device="cuda")
    seq_ids = torch.zeros(T, dtype=torch.long, device="cuda")
    bt = torch.zeros(1, 8, dtype=torch.long, device="cuda")
    bt[0, 0] = 1
    outs: list[torch.Tensor] = []
    with torch.no_grad():
        for layer in model.model.layers:
            cache = CacheEngine(
                cfg, CacheConfig(block_size=64), num_blocks=16, device="cuda", dtype=torch.bfloat16
            )
            x = layer(x, positions, seq_ids, [T], bt, cache, model.model.rope, False, None, False)
            outs.append(x.detach().float().cpu())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outs


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = tok.encode(PROMPT)
    print("ids:", ids)
    h = hf_layers(ids)
    c = cascade_layers(ids)
    for i, (a, b) in enumerate(zip(h, c)):
        d = (a - b).abs()
        print(f"layer {i:2d}: max|d|={d.max().item():.6f} mean|d|={d.mean().item():.6f}")


if __name__ == "__main__":
    main()
