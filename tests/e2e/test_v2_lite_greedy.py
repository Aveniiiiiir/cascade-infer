"""端到端验收（集群 GPU）：Cascade 贪心输出 vs HF transformers 逐 token 一致。

运行条件（缺一即 skip）:
- torch + CUDA
- 环境变量 CASCADE_E2E_MODEL 指向本地 DeepSeek-V2-Lite（HF 格式）目录
- transformers 可导入

用法（集群）:
    CASCADE_E2E_MODEL=/data/DeepSeek-V2-Lite pytest tests/e2e -q
"""

from __future__ import annotations

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


def _hf_greedy_all(model_dir: str, prompts: list[str], max_tokens: int) -> list[list[int]]:
    """HF 参考输出（模型只加载一次；逐条贪心）。"""
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    outs = []
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p)
            out = list(ids)
            for _ in range(max_tokens):
                x = torch.tensor([out], device="cuda")
                nxt = int(model(x).logits[0, -1].argmax().item())
                out.append(nxt)
                if nxt == tok.eos_token_id:
                    break
            outs.append(out)
    del model
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
