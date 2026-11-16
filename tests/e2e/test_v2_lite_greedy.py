"""端到端验收（集群 GPU）：Cascade 贪心输出 vs HF transformers 逐 token 一致。

运行条件（缺一即 skip）:
- torch + CUDA
- 环境变量 CASCADE_E2E_MODEL 指向本地 DeepSeek-V2-Lite（HF 格式）目录
- transformers 可导入

用法（集群）:
    CASCADE_E2E_MODEL=/data/DeepSeek-V2-Lite pytest tests/e2e -q
"""

from __future__ import annotations

import gc
import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("CASCADE_E2E_MODEL", "") == "",
        reason="CASCADE_E2E_MODEL 未设置（指向 HF 格式 V2-Lite 目录）",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA"),
]

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


def _hf_compat_patches() -> None:
    """让模型目录的官方 modeling_deepseek.py 在 transformers 4.5x 下可加载。

    官方实现是**权威参考**：它自带 YaRN rope（与 Cascade 公式一致）；
    transformers 内置实现会因 config 里 rope_scaling 的 int 类型校验而
    拒绝 YaRN（静默回退 plain rope），不能作为对拍基准。

    需要的三处兼容：
    1. flash_attn：官方 modeling 在 import 期硬检查；stub 一个低版本模块
       （显式传 attn_implementation="eager"，不会真正调用它）
    2. is_torch_fx_available：transformers 4.5x 已移除
    3. DynamicCache.get_usable_length：4.5x 已移除（我们传 use_cache=False，
       仅 import 期兜底）
    """
    import sys
    import types

    if "flash_attn" not in sys.modules:
        import importlib.machinery

        fa = types.ModuleType("flash_attn")
        fa.__version__ = "0.0.0"  # 低于 is_flash_attn_2_available 门槛
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


def _hf_greedy_all(model_dir: str, prompts: list[str], max_tokens: int) -> list[list[int]]:
    """HF 参考输出（模型只加载一次；逐条贪心）。

    设备可用 CASCADE_E2E_HF_DEVICE 覆盖（cuda / cpu）——显存紧张时把参考
    模型放 CPU（主机内存需 ≥ 模型大小，且很慢）。
    """
    _hf_compat_patches()
    import transformers

    hf_device = os.environ.get("CASCADE_E2E_HF_DEVICE", "cuda")
    tok = transformers.AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=hf_device,
        trust_remote_code=True,  # 官方 modeling_deepseek.py（自带 YaRN rope）
        attn_implementation="eager",  # 不用 flash-attn stub
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    outs = []
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p)
            out = list(ids)
            for _ in range(max_tokens):
                x = torch.tensor([out], device=hf_device)
                nxt = int(model(x, use_cache=False).logits[0, -1].argmax().item())
                out.append(nxt)
                if nxt == tok.eos_token_id:
                    break
            outs.append(out)
    del model
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    return outs


def test_greedy_tokens_match_hf() -> None:
    from cascade.entrypoints.llm import LLM
    from cascade.sampling_params import SamplingParams

    model_dir = os.environ["CASCADE_E2E_MODEL"]
    # --- Cascade：全部 prompt 混合批贪心生成（同时验证混合批组合不变）---
    llm = LLM(model_dir, max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=4096)
    tok = llm.tokenizer
    sp = SamplingParams(
        temperature=0.0, max_tokens=MAX_TOKENS, seq_seed=0, stop_token_ids=(llm.eos_token_id,)
    )
    prompt_ids = [tok.encode(p) for p in PROMPTS]
    results = llm._engine.generate(prompt_ids, sp)

    # **显存关键**：Cascade 权重（V2-Lite bf16 ≈ 29.3 GiB）与 KV 必须在上卡
    # HF 参考模型之前释放——两份模型同时驻留需 ~59 GiB，48GB 卡也放不下。
    # 释放后单模型峰值 ≈ 29.3 GiB 权重 + 上下文/激活，32GB 卡可跑（偏紧）。
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # --- HF 参考 ---
    hf_outs = _hf_greedy_all(model_dir, PROMPTS, MAX_TOKENS)

    mismatches = []
    for i, (p, hf_ids) in enumerate(zip(PROMPTS, hf_outs)):
        got = prompt_ids[i] + results[f"req-{i}"]
        if got != hf_ids:
            n = min(len(got), len(hf_ids))
            first = next((j for j in range(n) if got[j] != hf_ids[j]), n)
            mismatches.append((p, first, got[first : first + 8], hf_ids[first : first + 8]))
    assert not mismatches, (
        f"{len(mismatches)}/{len(PROMPTS)} 条 prompt 与 HF 不一致（prompt, 首个分歧位置, cascade, hf）: "
        f"{mismatches[:3]}"
    )
