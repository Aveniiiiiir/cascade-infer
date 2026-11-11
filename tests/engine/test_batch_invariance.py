"""引擎级批不变验收（集群 GPU，真实模型）：Phase 3 核心验收项。

DeepInfer 验证范式（tests/models/test_batch_invariant.py）：整批 generate
与单条运行（相同 seed）输出逐 token 一致 —— 采样层 RNG 只依赖
(seed, 绝对位置, 用途流)，kernel 归约序固定（无原子）由本测试端到端
兜底。

运行条件: CASCADE_E2E_MODEL + CUDA。
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("CASCADE_E2E_MODEL", "") == "",
        reason="CASCADE_E2E_MODEL 未设置",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA"),
]

PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):",
    "中国的首都是",
    "Water boils at",
    "In a shocking finding, scientists discovered",
    "for i in range(10):",
    "Machine learning is",
    "Translate to English: 早上好",
]


def _make_llm(model_dir: str):
    from cascade.entrypoints.llm import LLM

    return LLM(model_dir, max_model_len=2048, max_num_seqs=8, max_num_batched_tokens=2048)


def _params(llm, temperature: float, seed_base: int, n: int):
    from cascade.sampling_params import SamplingParams

    return [
        SamplingParams(
            temperature=temperature, max_tokens=48, seq_seed=seed_base + i,
            stop_token_ids=(llm.eos_token_id,),
        )
        for i in range(n)
    ]


def test_batched_equals_single_temperature_sampling() -> None:
    """温度采样（非贪心）：混批 generate ≡ 逐条独立运行（同 seed）。"""
    model_dir = os.environ["CASCADE_E2E_MODEL"]
    llm = _make_llm(model_dir)
    tok = llm.tokenizer
    ids = [tok.encode(p) for p in PROMPTS]

    # 整批（8 条混合批：prefill 与 decode 交错）
    batch = llm._engine.generate(ids, _params(llm, temperature=0.7, seed_base=100, n=len(ids)))

    # 逐条（batch=1）
    for i, pids in enumerate(ids):
        sp = _params(llm, temperature=0.7, seed_base=100, n=len(ids))[i]
        single = llm._engine.generate([pids], sp)
        got = single["req-0"]
        exp = batch[f"req-{i}"]
        assert got == exp, (
            f"prompt {i!r} 批/单不一致: 前 8 个分歧 "
            f"{[j for j in range(min(len(got), len(exp))) if got[j] != exp[j]][:8]}"
        )


def test_batched_equals_single_greedy() -> None:
    """贪心（eps 温度退化路径）：混批 ≡ 逐条。"""
    model_dir = os.environ["CASCADE_E2E_MODEL"]
    llm = _make_llm(model_dir)
    tok = llm.tokenizer
    ids = [tok.encode(p) for p in PROMPTS[:4]]
    batch = llm._engine.generate(ids, _params(llm, temperature=0.0, seed_base=7, n=4))
    for i, pids in enumerate(ids):
        sp = _params(llm, temperature=0.0, seed_base=7, n=4)[i]
        single = llm._engine.generate([pids], sp)
        assert single["req-0"] == batch[f"req-{i}"], f"greedy prompt {i} 批/单不一致"


def test_same_seed_reproducible() -> None:
    """同 seed 两次运行 bit 级一致（确定性采样）。"""
    model_dir = os.environ["CASCADE_E2E_MODEL"]
    llm = _make_llm(model_dir)
    tok = llm.tokenizer
    ids = [tok.encode(p) for p in PROMPTS[:4]]
    a = llm._engine.generate(ids, _params(llm, temperature=0.9, seed_base=42, n=4))
    b = llm._engine.generate(ids, _params(llm, temperature=0.9, seed_base=42, n=4))
    assert a == b
