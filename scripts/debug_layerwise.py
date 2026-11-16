"""逐层对拍：embedding / rope cos-sin / layer0 输出（Cascade vs 官方 HF）。"""

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
    x = torch.tensor([ids], device="cuda")
    captured = {}

    def hook_layer0(_m, inp, out):
        captured["layer0"] = (inp[0].detach().float().cpu(), (out[0] if isinstance(out, tuple) else out).detach().float().cpu())

    h = model.model.layers[0].register_forward_hook(hook_layer0)
    with torch.no_grad():
        out = model(x, use_cache=False)
    h.remove()
    emb = model.model.embed_tokens(x).detach().float().cpu()
    rope = model.model.layers[0].self_attn.rotary_emb
    cos = rope.cos_cached[:64].detach().float().cpu()
    logits = out.logits[0, -1].detach().float().cpu()
    print("HF rope attention factor check: cos[1,:4]", cos[1, :4].tolist())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"emb": emb, "layer0_in": captured["layer0"][0], "layer0_out": captured["layer0"][1], "cos": cos, "logits": logits}


def cascade_probe(ids: list[int]):
    from transformers import AutoTokenizer

    from cascade.config import CacheConfig
    from cascade.model_executor.model_loader import load_deepseek_v2
    from cascade.worker.cache_engine import CacheEngine

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model, cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    T = len(ids)
    x = torch.tensor(ids, device="cuda")
    with torch.no_grad():
        emb = model.model.embed_tokens(x)
        layer0_in = emb.detach().float().cpu()
        # 单层手动 forward（与模型内部相同的入参）
        from cascade.config import CacheConfig as CC

        cache = CacheEngine(cfg, CC(block_size=64), num_blocks=16, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(T, device="cuda")
        seq_ids = torch.zeros(T, dtype=torch.long, device="cuda")
        bt = torch.zeros(1, 8, dtype=torch.long, device="cuda")
        bt[0, 0] = 1
        h0 = model.model.layers[0](
            emb, positions, seq_ids, [T], bt, cache, model.model.rope, False, None, False
        )
        layer0_out = h0.detach().float().cpu()
        logits = model(x, positions, seq_ids, [T], bt, CacheEngine(cfg, CC(block_size=64), num_blocks=16, device="cuda", dtype=torch.bfloat16), use_absorbed=True)
        logits = logits[-1].detach().float().cpu()
        cos, _ = model.model.rope.cos_sin(64)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"emb": emb.detach().float().cpu(), "layer0_in": layer0_in, "layer0_out": layer0_out, "cos": cos.float().cpu(), "logits": logits}


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = tok.encode(PROMPT)
    print("ids:", ids)
    h = hf_probe(ids)
    c = cascade_probe(ids)
    for k in ("emb", "layer0_in", "layer0_out", "logits"):
        d = (h[k] - c[k]).abs()
        print(f"{k}: max|d|={d.max().item():.6f} mean|d|={d.mean().item():.6f}")
    dc = (h["cos"] - c["cos"]).abs()
    print(f"rope cos: max|d|={dc.max().item():.6f}")


if __name__ == "__main__":
    main()
