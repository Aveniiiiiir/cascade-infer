"""引擎级 MTP 端到端测试（torch；贪心验收：MTP ≡ 非 MTP 逐 token）。

tiny 随机模型 + MockDraftModel 走 LLMEngine 完整闭环（调度/前向/验证/
提交/草案发射）。贪心下任何管线 bug（行错位/草案错配/回滚错误）都会
破坏逐 token 一致性。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from cascade.engine.llm_engine import LLMEngine
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from cascade.model_executor.models.mtp_nextn import MockDraftModel
from cascade.sampling_params import SamplingParams


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        model_type="deepseek_v2",
        max_model_len=512,
        vocab_size=512,
        num_layers=3,
        hidden_size=64,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        num_kv_heads=1,
        intermediate_size=32,
        moe_intermediate_size=32,
        first_k_dense_replace=1,
        n_routed_experts=8,
        n_shared_experts=2,
        num_experts_per_tok=4,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        rms_norm_eps=1e-5,
    )


def _make_engine(device: str, with_mtp: bool, seed: int = 0, k: int = 3) -> LLMEngine:
    torch.manual_seed(seed)
    cfg = _tiny_cfg()
    engine_cfg = EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=512, nextn_draft_tokens=(k if with_mtp else 0)),
    )
    engine_cfg.validate()
    model = DeepseekV2ForCausalLM(cfg, device=device).to(device, torch.float32)
    draft = MockDraftModel(cfg.vocab_size, p_top1=0.6) if with_mtp else None
    return LLMEngine(
        engine_cfg, device=device, dtype=torch.float32,
        draft_model=draft, model=model, model_cfg=cfg,
    )


def _prompts() -> list[list[int]]:
    g = torch.Generator().manual_seed(42)
    return [torch.randint(1, 500, (length,), generator=g).tolist() for length in (11, 23, 7, 31)]


@pytest.mark.parametrize("device", _devices())
def test_greedy_mtp_equals_non_mtp(device: str) -> None:
    prompts = _prompts()
    sp = [SamplingParams(temperature=0.0, max_tokens=30, seq_seed=i) for i in range(len(prompts))]
    plain = _make_engine(device, with_mtp=False).generate(prompts, sp)
    mtp = _make_engine(device, with_mtp=True).generate(prompts, sp)
    for i in range(len(prompts)):
        assert mtp[f"req-{i}"] == plain[f"req-{i}"], (
            f"prompt {i}: MTP 与非 MTP 贪心输出不一致 "
            f"(首分歧 {[j for j in range(min(len(mtp[f'req-{i}']), len(plain[f'req-{i}']))) if mtp[f'req-{i}'][j] != plain[f'req-{i}'][j]][:8]})"
        )


@pytest.mark.parametrize("device", _devices())
def test_mtp_multi_prompt_mixed_batch(device: str) -> None:
    """多请求混合批（prefill 与 MTP decode 交错）贪心一致。"""
    prompts = _prompts()
    sp = [SamplingParams(temperature=0.0, max_tokens=40, seq_seed=7 + i) for i in range(len(prompts))]
    plain = _make_engine(device, with_mtp=False, seed=3).generate(prompts, sp)
    mtp = _make_engine(device, with_mtp=True, seed=3).generate(prompts, sp)
    assert mtp == plain


@pytest.mark.parametrize("device", _devices())
def test_mtp_chunked_prefill_coexistence(device: str) -> None:
    """chunked prefill（小预算）与 MTP decode 共存：贪心一致。"""
    cfg = _tiny_cfg()
    engine_cfg = EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16, nextn_draft_tokens=2),
    )
    engine_cfg.validate()
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(cfg, device=device).to(device, torch.float32)
    draft = MockDraftModel(cfg.vocab_size)
    mtp_engine = LLMEngine(
        engine_cfg, device=device, dtype=torch.float32,
        draft_model=draft, model=model, model_cfg=cfg,
    )
    prompts = _prompts()[:2]
    sp = [SamplingParams(temperature=0.0, max_tokens=25, seq_seed=1) for _ in prompts]
    got = mtp_engine.generate(prompts, sp)
    # 对照（同权重：重建时重新 seed）
    torch.manual_seed(0)
    model2 = DeepseekV2ForCausalLM(cfg, device=device).to(device, torch.float32)
    plain_cfg = EngineConfig(
        model=cfg, cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16),
    )
    plain_cfg.validate()
    plain = LLMEngine(plain_cfg, device=device, dtype=torch.float32, model=model2, model_cfg=cfg)
    exp = plain.generate(prompts, sp)
    assert got == exp
