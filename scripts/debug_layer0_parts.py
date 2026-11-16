"""layer0 内部拆分对拍：attention 输出 / MLP 输出（Cascade vs HF）。"""

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


def hf_probe(ids: list[int]):
    _hf_patches()
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    layer0 = model.model.layers[0]
    cap = {}

    def h_attn(_m, _i, o):
        cap["attn"] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()

    def h_mlp(_m, _i, o):
        cap["mlp"] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()

    def h_ln_in(_m, i, _o):
        cap["ln_in"] = i[0].detach().float().cpu()

    h1 = layer0.self_attn.register_forward_hook(h_attn)
    h2 = layer0.mlp.register_forward_hook(h_mlp)
    h3 = layer0.input_layernorm.register_forward_hook(h_ln_in)
    with torch.no_grad():
        model(torch.tensor([ids], device="cuda"), use_cache=False)
    for h in (h1, h2, h3):
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cap


def cascade_probe(ids: list[int]):
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
    layer0 = model.model.layers[0]
    cap = {}
    with torch.no_grad():
        ln = layer0.input_layernorm(x)
        cap["ln_in"] = ln.detach().float().cpu()
        cache = CacheEngine(
            cfg, CacheConfig(block_size=64), num_blocks=16, device="cuda", dtype=torch.bfloat16
        )
        a = layer0.self_attn(
            ln, positions, seq_ids, [T], bt, cache, model.model.rope, False, None, False
        )
        cap["attn"] = a.detach().float().cpu()
        h = x + a
        mlp_out = layer0.mlp(layer0.post_attention_layernorm(h))
        cap["mlp"] = mlp_out.detach().float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cap


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = tok.encode(PROMPT)
    h = hf_probe(ids)
    c = cascade_probe(ids)
    for k in ("ln_in", "attn", "mlp"):
        d = (h[k] - c[k]).abs()
        print(f"layer0 {k}: max|d|={d.max().item():.6f} mean|d|={d.mean().item():.6f}")


if __name__ == "__main__":
    main()
