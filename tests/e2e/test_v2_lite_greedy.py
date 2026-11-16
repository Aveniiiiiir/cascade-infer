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
    margins: list[list[float]] = []
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p)
            out = list(ids)
            ms: list[float] = []
            for _ in range(max_tokens):
                x = torch.tensor([out], device=hf_device)
                logits = model(x, use_cache=False).logits[0, -1]
                top2 = logits.topk(2)
                nxt = int(top2.indices[0])
                ms.append(float(top2.values[0] - top2.values[1]))
                out.append(nxt)
                if nxt == tok.eos_token_id:
                    break
            outs.append(out)
            margins.append(ms)
    del model
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    return outs, margins


def _cascade_teacher_forcing(
    llm, seqs: list[list[int]], prompt_lens: list[int], use_absorbed: bool
):
    """在 HF 的完整 token 序列上前向，逐位置比较 argmax 与「HF 的下一步 token」。

    只比较**生成段**（p ≥ prompt_len-1）：prompt 内的 token 并非 HF 的
    argmax（人类文本），比较它们没有意义。

    Returns: (match, total, mismatches) —— mismatches: [(prompt_idx, pos, pred, target)]
    """
    import torch as _torch

    from cascade.config import CacheConfig
    from cascade.worker.cache_engine import CacheEngine

    runner = llm._engine.runner
    model = runner.model
    cfg = runner.cfg
    match = total = 0
    mismatches = []
    with _torch.no_grad():
        for i, seq in enumerate(seqs):
            T = len(seq)
            cache = CacheEngine(
                cfg, CacheConfig(block_size=64), num_blocks=64,
                device="cuda", dtype=_torch.bfloat16,
            )
            bt = _torch.zeros(1, 4, dtype=_torch.long, device="cuda")
            bt[0, 0] = 1
            logits = model(
                _torch.tensor(seq, device="cuda"),
                _torch.arange(T, device="cuda"),
                _torch.zeros(T, dtype=_torch.long, device="cuda"),
                [T], bt, cache, use_absorbed=use_absorbed,
            ).float()
            for p in range(prompt_lens[i] - 1, T - 1):
                pred = int(logits[p].argmax())
                target = seq[p + 1]
                total += 1
                if pred == target:
                    match += 1
                else:
                    mismatches.append((i, p, pred, target))
    return match, total, mismatches


def test_greedy_tokens_match_hf() -> None:
    """e2e 验收：teacher-forcing 逐位置 argmax 为主判据。

    bf16 下 free-running 32 步不可能逐 token 完全一致：单步 logits 的
    tie 级数值差（margin < 0.1）会在自回归中放大成分叉。因此：
    - 主判据：在 HF 的 token 序列上前向（teacher forcing），逐位置 argmax
      一致率 ≥ 99.5%（explicit）/ ≥ 98%（absorbed kernel），分歧仅允许
      出现在 HF 该步 margin 极小的位置；
    - 辅助：free-running（混合批，验证批不变）每条 prompt 的**首次分歧**
      也必须发生在 HF margin 极小的位置。
    """
    from cascade.entrypoints.llm import LLM
    from cascade.sampling_params import SamplingParams

    model_dir = os.environ["CASCADE_E2E_MODEL"]
    explicit_rate_min = 0.995
    absorbed_rate_min = 0.98
    margin_tol = 0.25

    # --- HF 参考（先跑并释放：两份 29GB 模型不同时驻留）---
    hf_outs, hf_margins = _hf_greedy_all(model_dir, PROMPTS, MAX_TOKENS)

    # --- Cascade：混合批贪心生成（验证批组合不变）---
    llm = LLM(model_dir, max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=4096)
    tok = llm.tokenizer
    sp = SamplingParams(
        temperature=0.0, max_tokens=MAX_TOKENS, seq_seed=0, stop_token_ids=(llm.eos_token_id,)
    )
    prompt_ids = [tok.encode(p) for p in PROMPTS]
    results = llm._engine.generate(prompt_ids, sp)

    # --- teacher forcing（复用已加载模型，显式 / 吸收两条路径）---
    tf_stats = {}
    for name, absorbed in (("explicit", False), ("absorbed", True)):
        m, n, bad = _cascade_teacher_forcing(llm, hf_outs, [len(x) for x in prompt_ids], absorbed)
        tf_stats[name] = (m, n, bad)
        rate = m / n
        min_rate = explicit_rate_min if name == "explicit" else absorbed_rate_min
        assert rate >= min_rate, (
            f"teacher-forcing {name}: argmax 一致率 {m}/{n} = {rate:.4f} < {min_rate}；"
            f"前 5 处分歧 {bad[:5]}"
        )
        # 分歧只允许出现在 HF margin 极小的 tie 级位置
        bad_margins = []
        for (pi, pos, pred, target) in bad:
            step = pos - len(prompt_ids[pi]) + 1
            margin = hf_margins[pi][step] if 0 <= step < len(hf_margins[pi]) else float("inf")
            bad_margins.append((pi, pos, pred, target, margin))
        assert all(b[4] < margin_tol for b in bad_margins), (
            f"teacher-forcing {name}: 存在非 tie 级分歧 {bad_margins[:5]}"
        )

    # --- free-running：首次分歧必须在 tie 级位置 ---
    mismatches = []
    for i, p in enumerate(PROMPTS):
        got = prompt_ids[i] + results[f"req-{i}"]
        hf_ids = hf_outs[i]
        n = min(len(got), len(hf_ids))
        first = next((j for j in range(n) if got[j] != hf_ids[j]), None)
        if first is not None:
            step = first - len(prompt_ids[i])
            margin = hf_margins[i][step] if 0 <= step < len(hf_margins[i]) else float("inf")
            mismatches.append((p, first, margin, got[first : first + 4], hf_ids[first : first + 4]))
        assert first is None or margin < margin_tol, (
            f"free-running '{p}': 首分歧位置 {first} 处 HF margin={margin}（非 tie 级）"
        )
    assert len(mismatches) <= 10, (
        f"{len(mismatches)}/{len(PROMPTS)} 条 prompt 分叉（bf16 累积噪声上限 10（实测 6-7））：{mismatches[:3]}"
    )
