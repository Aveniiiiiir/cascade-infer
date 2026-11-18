"""LLMEngine 必须无条件 validate：纯 LLM 主路径 runtime_max_model_len 曾恒为 0。

2026-11-18 真实模型 HTTP 冒烟暴露：LLMEngine.__init__ 此前只在 draft_model
分支调用 engine_cfg.validate()，纯 LLM 主路径（生产入口）从不调用 →
runtime_max_model_len 保持 dataclass 默认 0，后果链：
- _profile_num_blocks 的 max_useful = (max_num_seqs*0 + bs-1)//bs + 1 = 1
  → KV 池被 cap 到 1 块（1.63GiB 预算实测只分到 1 块，并发尽毁）
- _prewarm_rope(model, 0) → cos/sin 只有 1024 位置（>1024 的前向越界）
- CudaGraphRunner.max_bt = 0//bs + 2 = 2 列 block 表（长序列 decode 越界）
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from cascade.engine.llm_engine import LLMEngine
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        model_type="deepseek_v2", max_model_len=256, vocab_size=512, num_layers=2,
        hidden_size=64, num_attention_heads=4, q_lora_rank=32, kv_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16, num_kv_heads=1,
        intermediate_size=32, moe_intermediate_size=32, first_k_dense_replace=1,
        n_routed_experts=8, n_shared_experts=2, num_experts_per_tok=4,
        scoring_func="sigmoid", topk_method="noaux_tc", rms_norm_eps=1e-5,
    )


def test_engine_unconditional_validate() -> None:
    """不手动 validate 的纯 LLM 主路径：引擎构造后 runtime/位置上界必须正确。

    复现口径：调用方只构造 EngineConfig（不调 validate）直接传入——
    LLM() 门面即此用法。
    """
    cfg = _tiny_cfg()
    engine_cfg = EngineConfig(
        model=cfg, cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=256),
    )
    # dataclass 默认值：构造后未计算
    assert engine_cfg.runtime_max_model_len == 0

    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(cfg, device="cpu").to("cpu", torch.float32)
    engine = LLMEngine(
        engine_cfg, device="cpu", dtype=torch.float32, model=model, model_cfg=cfg,
    )
    # 无 draft（K=0）→ runtime = max_model_len + 1
    assert engine.cfg.runtime_max_model_len == cfg.max_model_len + 1
    # KV 池不再被 cap 到 1：CPU 默认预算 256MiB 下应有远多于 1 的块
    assert engine.runner.cache.num_blocks > 1
    # rope 预热到位（_max_cached 至少覆盖 runtime 位置）
    rope = engine.runner.model.model.rope
    assert rope._max_cached >= engine.cfg.runtime_max_model_len


def test_engine_unconditional_validate_with_draft() -> None:
    """draft 路径：K 计入位置余量（max + K + 1）不被无条件 validate 破坏。"""
    from tests.engine.test_engine_mtp import MockDraftModel  # 复用 mock

    cfg = _tiny_cfg()
    k = 3
    engine_cfg = EngineConfig(
        model=cfg, cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=256, nextn_draft_tokens=k),
    )
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(cfg, device="cpu").to("cpu", torch.float32)
    engine = LLMEngine(
        engine_cfg, device="cpu", dtype=torch.float32, model=model, model_cfg=cfg,
        draft_model=MockDraftModel(cfg.vocab_size),
    )
    # nextn 层 = 0（V2 无 MTP 层）→ runtime = max + 2*0 + K + 1
    assert engine.cfg.runtime_max_model_len == cfg.max_model_len + k + 1


__all__ = ["test_engine_unconditional_validate", "test_engine_unconditional_validate_with_draft"]
