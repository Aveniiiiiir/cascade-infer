"""attention 内部对拍：q/kv 投影、rope cos/sin、rope 后 q/k、attention 输出。"""

from __future__ import annotations

import gc
import os
import sys

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")


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


def fixed_x(hidden: int = 2048, rows: int = 6) -> torch.Tensor:
    vals = torch.arange(rows * hidden, dtype=torch.float32) / 1000.0 - 3.0
    return vals.reshape(rows, hidden).to(torch.bfloat16).cuda()


def hf_probe(x: torch.Tensor):
    _hf_patches()
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    attn = model.model.layers[0].self_attn
    cap = {}

    def h_q(_m, _i, o):
        cap["q"] = o.detach().float().cpu()

    def h_kv(_m, _i, o):
        cap["kv_b"] = o.detach().float().cpu()

    def h_rope(_m, _i, o):
        cos, sin = o
        cap["cos"] = cos.detach().float().cpu()
        cap["sin"] = sin.detach().float().cpu()

    def h_attn(_m, _i, o):
        cap["attn_out"] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()

    def h_cproj(_m, _i, o):
        cap["c_raw"] = o.detach().float().cpu()

    def h_cln(_m, i, o):
        cap["ln_in"] = i[0].detach().float().cpu()

    hooks = [
        attn.q_proj.register_forward_hook(h_q),
        attn.kv_b_proj.register_forward_hook(h_kv),
        attn.rotary_emb.register_forward_hook(h_rope),
        attn.register_forward_hook(h_attn),
        attn.kv_a_proj_with_mqa.register_forward_hook(h_cproj),
        attn.kv_a_layernorm.register_forward_hook(h_cln),
    ]
    L = x.shape[0]
    mask = torch.full((1, 1, L, L), torch.finfo(torch.bfloat16).min, device="cuda")
    mask = torch.triu(mask, diagonal=1)
    pos = torch.arange(L, device="cuda").unsqueeze(0)
    with torch.no_grad():
        attn(x.unsqueeze(0), attention_mask=mask, position_ids=pos, past_key_value=None, use_cache=False)
    for h in hooks:
        h.remove()
    # 补：rope 前 k_pe / q_pe 由 q/kv 投影拆分
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cap


def cascade_probe(x: torch.Tensor):
    from cascade.config import CacheConfig
    from cascade.model_executor.model_loader import load_deepseek_v2
    from cascade.worker.cache_engine import CacheEngine

    model, cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    attn = model.model.layers[0].self_attn
    rope = model.model.rope
    L = x.shape[0]
    pos = torch.arange(L, device="cuda")
    cap = {}
    with torch.no_grad():
        q = attn.q_proj(x)
        qv = q.view(L, attn.num_heads, attn.qk_nope + attn.qk_rope)
        q_nope, q_pe = qv[..., : attn.qk_nope], qv[..., attn.qk_nope :]
        c_raw = attn.kv_a_proj_with_mqa(x)
        c_norm = attn.kv_a_layernorm(c_raw[..., : attn.kv_lora])
        k_pe = c_raw[..., attn.kv_lora :]
        q_pe2, k_pe2 = rope.apply(q_pe, k_pe, pos)
        cos, sin = rope.cos_sin(L)
        kv_b = attn.kv_b_proj(c_norm)
        cap.update(
            q=q.detach().float().cpu(),
            c_raw=c_raw.detach().float().cpu(),
            ln_in=c_norm.detach().float().cpu(),
            kv_b=kv_b.detach().float().cpu(),
            cos=cos.float().cpu(),
            sin=sin.float().cpu(),
        )
        cache = CacheEngine(
            cfg, CacheConfig(block_size=64), num_blocks=16, device="cuda", dtype=torch.bfloat16
        )
        bt = torch.zeros(1, 8, dtype=torch.long, device="cuda")
        bt[0, 0] = 1
        out = attn(x, pos, torch.zeros(L, dtype=torch.long, device="cuda"), [L], bt, cache, rope, False, None, False)
        cap["attn_out"] = out.detach().float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cap


def main() -> None:
    x = fixed_x()
    h = hf_probe(x)
    c = cascade_probe(x)
    print("keys hf:", sorted(h), "| cascade:", sorted(c))
    for k in ("q", "c_raw", "ln_in", "kv_b", "cos", "sin", "attn_out"):
        if k in h and k in c:
            d = (h[k] - c[k]).abs()
            print(f"{k}: max|d|={d.max().item():.6f} mean|d|={d.mean().item():.6f}")


if __name__ == "__main__":
    main()
