"""隔离 rope：固定 q/k 输入，Cascade rope.apply vs HF 官方 rope+apply_rotary_pos_emb。"""

from __future__ import annotations

import gc
import os
import sys

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")
H, ROPE, L = 16, 64, 6


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


def fixed_inputs():
    q = (torch.arange(L * H * ROPE, dtype=torch.float32).reshape(L, H, ROPE) / 500.0 - 2.0)
    k = (torch.arange(L * ROPE, dtype=torch.float32).reshape(L, ROPE) / 300.0 - 1.0)
    pos = torch.arange(L, dtype=torch.long)
    return q.to(torch.bfloat16).cuda(), k.to(torch.bfloat16).cuda(), pos.cuda()


def hf_rope(q, k, pos):
    _hf_patches()
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    mod_name = next(m for m in sys.modules if m.endswith("modeling_deepseek"))
    apply_rotary = sys.modules[mod_name].apply_rotary_pos_emb
    a = model.model.layers[0].self_attn
    with torch.no_grad():
        cos, sin = a.rotary_emb(k, seq_len=L)
        q4 = q.transpose(0, 1).unsqueeze(0)  # [1,H,L,64]
        k4 = k.view(1, L, 1, ROPE).transpose(1, 2)  # [1,1,L,64]
        q2, k2 = apply_rotary(q4, k4, cos, sin, pos.unsqueeze(0))
        cos_c, sin_c = cos.detach().float().cpu(), sin.detach().float().cpu()
    out = (q2.float().cpu(), k2.float().cpu(), cos_c, sin_c)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def cascade_rope(q, k, pos):
    from cascade.model_executor.model_loader import load_deepseek_v2

    model, _cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    rope = model.model.rope
    with torch.no_grad():
        q2, k2 = rope.apply(q, k, pos)
        cos, sin = rope.cos_sin(L)
    out = (
        q2.transpose(0, 1).unsqueeze(0).float().cpu(),
        k2.view(1, 1, L, ROPE).float().cpu(),
        cos.float().cpu(),
        sin.float().cpu(),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    q, k, pos = fixed_inputs()
    h = hf_rope(q, k, pos)
    c = cascade_rope(q, k, pos)
    for name, a, b in zip(("q@rope", "k@rope", "cos", "sin"), h, c):
        d = (a - b).abs()
        print(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)} max|d|={d.max().item():.6f}")
    print("q input[0,0,:4]:", q[0, 0, :4].tolist())
    print("hf q2[0,0,0,:4]:", h[0][0, 0, 0, :4].tolist())
    print("cs q2[0,0,0,:4]:", c[0][0, 0, 0, :4].tolist())


if __name__ == "__main__":
    main()
