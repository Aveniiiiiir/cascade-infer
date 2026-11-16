"""attention 逐步对拍：q/k 投影 → rope → scores → softmax → out。"""

from __future__ import annotations

import gc
import os
import sys

import torch

MODEL_DIR = os.environ.get("CASCADE_E2E_MODEL", "/root/autodl-tmp/DeepSeek-V2-Lite")
H, NOPE, ROPE, VL, KV = 16, 128, 64, 128, 512


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


def hf_manual(x: torch.Tensor):
    _hf_patches()
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    mod_name = next(m for m in sys.modules if m.endswith("modeling_deepseek"))
    apply_rotary_pos_emb = sys.modules[mod_name].apply_rotary_pos_emb
    a = model.model.layers[0].self_attn
    L = x.shape[0]
    pos = torch.arange(L, device="cuda").unsqueeze(0)
    with torch.no_grad():
        q = a.q_proj(x).view(1, L, H, NOPE + ROPE).transpose(1, 2)
        qn, qp = q[..., :NOPE], q[..., NOPE:]
        kvr = a.kv_a_proj_with_mqa(x)
        c, kpe = kvr[..., :KV], kvr[..., KV:]
        c_n = a.kv_a_layernorm(c)
        kvb = a.kv_b_proj(c_n).view(1, L, H, NOPE + VL).transpose(1, 2)
        kn, v = kvb[..., :NOPE], kvb[..., NOPE:]
        kpe = kpe.view(1, L, 1, ROPE).transpose(1, 2)
        cos, sin = a.rotary_emb(v, seq_len=L)
        # 注意 pos 已是 [1, L]，传给 apply_rotary_pos_emb 时不要再 unsqueeze
        # （多加一维会让 cos 变 5D，静默广播出错误结果）
        qp2, kpe2 = apply_rotary_pos_emb(qp, kpe, cos, sin, pos)
        query = torch.cat((qn, qp2), dim=-1)
        key = torch.cat((kn, kpe2.expand(1, H, L, ROPE)), dim=-1)
        scores = torch.matmul(query, key.transpose(2, 3)) * a.softmax_scale
        mask = torch.triu(
            torch.full((1, 1, L, L), torch.finfo(torch.bfloat16).min, device="cuda"), diagonal=1
        )
        scores_m = scores + mask
        w = torch.softmax(scores_m, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(w, v)
    res = dict(
        scores=scores.float().cpu(),
        w=w.float().cpu(),
        out=out.float().cpu(),
        qp_pre=qp.float().cpu(),
        qp2=qp2.float().cpu(),
        kpe2=kpe2.float().cpu(),
        scale=a.softmax_scale,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def cascade_manual(x: torch.Tensor):
    from cascade.model_executor.model_loader import load_deepseek_v2

    model, cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    model.eval()
    a = model.model.layers[0].self_attn
    rope = model.model.rope
    L = x.shape[0]
    pos = torch.arange(L, device="cuda")
    with torch.no_grad():
        qn, qp = a.project_q(x)
        c, kpe = a.project_kv(x)
        qp2, kpe2 = rope.apply(qp, kpe, pos)
        k_full = a.kv_b_proj(c).view(L, H, NOPE + VL)
        kn, v = k_full[..., :NOPE], k_full[..., NOPE:]
        query = torch.cat((qn, qp2), dim=-1).transpose(0, 1).unsqueeze(0)
        key = torch.cat((kn, kpe2.unsqueeze(1).expand(L, H, ROPE)), dim=-1).transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(query, key.transpose(2, 3)) * a.scale
        m = pos.unsqueeze(1) >= torch.arange(L, device="cuda").unsqueeze(0)
        scores_m = scores.masked_fill(~m.unsqueeze(0).unsqueeze(0), float("-inf"))
        w = torch.softmax(scores_m, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(w, v.transpose(0, 1).unsqueeze(0))
    res = dict(
        scores=scores.float().cpu(),
        w=w.float().cpu(),
        out=out.float().cpu(),
        qp_pre=qp.transpose(0, 1).unsqueeze(0).float().cpu(),
        qp2=qp2.transpose(0, 1).unsqueeze(0).float().cpu(),
        kpe2=kpe2.view(1, 1, L, ROPE).float().cpu(),
        scale=a.scale,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    x = fixed_x()
    h = hf_manual(x)
    c = cascade_manual(x)
    print("scale hf/cascade:", h["scale"], c["scale"])
    for k in ("qp_pre", "qp2", "kpe2", "scores", "w", "out"):
        a, b = h[k], c[k]
        if a.shape != b.shape:
            print(f"{k}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        d = (a - b).abs()
        print(f"{k}: max|d|={d.max().item():.6f} mean|d|={d.mean().item():.6f}")
    print("hf qp2[0,0,0,:6]:", h["qp2"][0, 0, 0, :6].tolist())
    print("cs qp2[0,0,0,:6]:", c["qp2"][0, 0, 0, :6].tolist())
    print("hf qp2[0,0,1,:6]:", h["qp2"][0, 0, 1, :6].tolist())
    print("cs qp2[0,0,1,:6]:", c["qp2"][0, 0, 1, :6].tolist())


if __name__ == "__main__":
    main()
