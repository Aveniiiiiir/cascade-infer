"""input_layernorm 对拍：weight / eps / 固定输入下的输出。"""

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


def main() -> None:
    outs = {}
    # --- Cascade ---
    from cascade.model_executor.model_loader import load_deepseek_v2

    model, _cfg = load_deepseek_v2(MODEL_DIR, device="cuda", dtype=torch.bfloat16)
    ln = model.model.layers[0].input_layernorm
    x = fixed_x()
    with torch.no_grad():
        outs["c_w"] = ln.weight.detach().float().cpu()
        outs["c_out"] = ln(x).detach().float().cpu()
        outs["c_eps"] = ln.variance_epsilon
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- HF ---
    _hf_patches()
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager",
    )
    hf.eval()
    ln_h = hf.model.layers[0].input_layernorm
    with torch.no_grad():
        outs["h_w"] = ln_h.weight.detach().float().cpu()
        outs["h_out"] = ln_h(x).detach().float().cpu()
        outs["h_eps"] = ln_h.variance_epsilon
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    print("eps cascade/hf:", outs["c_eps"], outs["h_eps"])
    dw = (outs["c_w"] - outs["h_w"]).abs()
    print(f"weight: max|d|={dw.max().item():.6f} mean|d|={dw.mean().item():.6f}")
    do = (outs["c_out"] - outs["h_out"]).abs()
    print(f"out: max|d|={do.max().item():.6f} mean|d|={do.mean().item():.6f}")
    print("c_out[0,:4]:", outs["c_out"][0, :4].tolist())
    print("h_out[0,:4]:", outs["h_out"][0, :4].tolist())


if __name__ == "__main__":
    main()
