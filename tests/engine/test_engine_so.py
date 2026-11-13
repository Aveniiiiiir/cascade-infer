"""引擎级 SO 测试（torch；stub 文法）：约束生效、MTP 共存、失败隔离。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from cascade.engine.llm_engine import LLMEngine
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from cascade.model_executor.models.mtp_nextn import MockDraftModel
from cascade.sampling_params import SamplingParams
from tests.engine.test_engine_mtp import _tiny_cfg


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_engine(device: str, with_mtp: bool = False, seed: int = 0) -> LLMEngine:
    torch.manual_seed(seed)
    cfg = _tiny_cfg()
    engine_cfg = EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(
            max_num_seqs=8, max_num_batched_tokens=256,
            nextn_draft_tokens=(3 if with_mtp else 0),
        ),
    )
    engine_cfg.validate()
    model = DeepseekV2ForCausalLM(cfg, device=device).to(device, torch.float32)
    draft = MockDraftModel(cfg.vocab_size) if with_mtp else None
    return LLMEngine(
        engine_cfg, device=device, dtype=torch.float32,
        draft_model=draft, model=model, model_cfg=cfg,
    )


def _prompts() -> list[list[int]]:
    g = torch.Generator().manual_seed(11)
    return [torch.randint(1, 500, (length,), generator=g).tolist() for length in (13, 21)]


@pytest.mark.parametrize("device", _devices())
def test_so_constrains_greedy_output(device: str) -> None:
    """stub 文法 "30,11,42"：贪心输出前三 token 必须逐位吻合；terminated
    后全开（后续为模型 argmax），max_tokens 收尾。"""
    from cascade.structured_output.compiler import has_xgrammar

    if has_xgrammar():
        pytest.skip("本测试依赖 StubGrammar 的逐步约束语义（无 xgrammar 环境）；"
                    "xgrammar 会把 '30,11,42' 当 regex 编译，语义不同")
    eng = _make_engine(device)
    schema = "30,11,42"
    prompts = _prompts()
    sp = [
        SamplingParams(temperature=0.0, max_tokens=10, seq_seed=i, so_args={"schema_type": "regex", "schema": schema})
        for i in range(len(prompts))
    ]
    out = eng.generate(prompts, sp)
    for i in range(len(prompts)):
        got = out[f"req-{i}"]
        assert got[:3] == [30, 11, 42], f"prompt {i}: SO 约束未生效 {got[:3]}"
        assert len(got) == 10


@pytest.mark.parametrize("device", _devices())
def test_so_with_mtp(device: str) -> None:
    """MTP + SO：提交的 token 序列仍满足约束（试探掩码 + 回滚正确）。"""
    from cascade.structured_output.compiler import has_xgrammar

    if has_xgrammar():
        pytest.skip("依赖 StubGrammar 语义（无 xgrammar 环境）")
    eng = _make_engine(device, with_mtp=True)
    schema = "7,19"
    prompts = _prompts()[:1]
    sp = [
        SamplingParams(temperature=0.0, max_tokens=8, seq_seed=5, so_args={"schema_type": "ebnf", "schema": schema})
        for _ in prompts
    ]
    out = eng.generate(prompts, sp)
    got = out["req-0"]
    assert got[:2] == [7, 19]
    assert len(got) == 8


@pytest.mark.parametrize("device", _devices())
def test_so_failure_isolation(device: str) -> None:
    """坏 schema 只判死本请求（FINISHED_SO_FAILED），其他请求正常完成。"""
    from cascade.structured_output.compiler import has_xgrammar

    eng = _make_engine(device)
    prompts = _prompts()
    good = SamplingParams(temperature=0.0, max_tokens=6, seq_seed=1)
    if has_xgrammar():
        pytest.skip("本机装了 xgrammar（无失败路径可测）")
    bad = SamplingParams(
        temperature=0.0, max_tokens=6, seq_seed=2,
        so_args={"schema_type": "json", "schema": {"type": "object"}},
    )
    ids = [eng.add_request(prompts[0], bad), eng.add_request(prompts[1], good)]
    results: dict[str, list[int]] = {i: [] for i in ids}
    finish: dict[str, str] = {}
    while eng.scheduler.has_unfinished_requests or eng._so_staging:
        for ro in eng.step():
            results[ro.request_id].extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                finish[ro.request_id] = ro.finish_reason
    # 冲刷（同步序下判死回执经 flush_final_outputs；staged 编译失败的
    # FINISHED_SO_FAILED 输出在 step() 顶部冲刷）
    for ro in eng.scheduler.flush_final_outputs():
        results[ro.request_id].extend(ro.new_token_ids)
        if ro.finish_reason is not None:
            finish[ro.request_id] = ro.finish_reason
    assert finish[ids[0]] == "so_failed"
    assert finish[ids[1]] == "length"
    assert len(results[ids[1]]) == 6
