"""引擎级 n-gram（无 draft 模型）投机解码端到端测试（torch）。

验收目标与 MTP 相同：**贪心输出与非投机逐 token 一致**（拒绝采样保证
输出分布不变），并验证：
- 草案模型确实被调用且拿到完整上下文（prompt‖已生成）
- 重复 prompt 的后缀匹配产生真实草案（token >= 0），不是全哨兵
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from cascade.engine.llm_engine import LLMEngine
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from cascade.sampling.ngram_draft import NGramDraftModel
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


class _RecordingNGram(NGramDraftModel):
    """记录调用与最近一次草案（断言集成路径真的跑起来了）。"""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.calls = 0
        self.saw_real_draft = False
        self.last_tokens: list[list[int]] = []
        self.last_contexts: list[list[int]] = []

    def draft_tokens(self, *args: object, **kwargs: object):  # type: ignore[override]
        self.calls += 1
        out = super().draft_tokens(*args, **kwargs)  # type: ignore[arg-type]
        self.last_tokens = [list(row) for row in out[0]]
        self.saw_real_draft = self.saw_real_draft or any(
            t >= 0 for row in self.last_tokens for t in row
        )
        self.last_contexts = [list(ctx) for ctx in kwargs.get("context_tokens", [])]  # type: ignore[union-attr]
        return out


def _make_engine(device: str, with_ngram: bool, seed: int = 0, k: int = 3) -> tuple[LLMEngine, object]:
    torch.manual_seed(seed)
    cfg = _tiny_cfg()
    engine_cfg = EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=512,
            nextn_draft_tokens=(k if with_ngram else 0),
        ),
    )
    engine_cfg.validate()
    model = DeepseekV2ForCausalLM(cfg, device=device).to(device, torch.float32)
    draft = _RecordingNGram(min_gram=2, max_gram=3, max_draft=k) if with_ngram else None
    engine = LLMEngine(
        engine_cfg, device=device, dtype=torch.float32,
        draft_model=draft, model=model, model_cfg=cfg,
    )
    return engine, draft


@pytest.mark.parametrize("device", _devices())
def test_greedy_ngram_equals_plain(device: str) -> None:
    prompts = [
        [7, 8, 9] * 8,  # 周期重复：草案必命中
        list(range(11, 34)),
        [5, 4, 3, 2] * 5,
        list(range(100, 124)),
    ]
    sp = [SamplingParams(temperature=0.0, max_tokens=24, seq_seed=i) for i in range(len(prompts))]
    plain, _ = _make_engine(device, with_ngram=False)
    ngram, draft = _make_engine(device, with_ngram=True)
    out_plain = plain.generate(prompts, sp)
    out_ngram = ngram.generate(prompts, sp)
    assert out_ngram == out_plain
    assert draft is not None and draft.calls > 0, "n-gram 草案从未发射"
    # 周期 prompt 的后缀命中 → 至少存在一个真实草案 token（非 -1 哨兵）
    assert draft.saw_real_draft, f"草案全为哨兵（上下文匹配失效）：{draft.last_tokens}"
    # 上下文包含 prompt‖已生成（首步 ≥ 最短 prompt 长度）
    assert all(len(c) >= 20 for c in draft.last_contexts)


@pytest.mark.parametrize("device", _devices())
def test_ngram_mixed_batch_chunked_prefill(device: str) -> None:
    """小预算 chunked prefill 与 n-gram 验证行共存：贪心一致。"""
    torch.manual_seed(4)
    cfg = _tiny_cfg()
    engine_cfg = EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16, nextn_draft_tokens=2),
    )
    engine_cfg.validate()
    model = DeepseekV2ForCausalLM(cfg, device=device).to(device, torch.float32)
    draft = _RecordingNGram(min_gram=2, max_gram=2, max_draft=2)
    eng = LLMEngine(engine_cfg, device=device, dtype=torch.float32, draft_model=draft, model=model, model_cfg=cfg)
    prompts = [[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], [2, 7, 1, 8, 2, 8, 1, 8, 2, 8]]
    sp = [SamplingParams(temperature=0.0, max_tokens=20, seq_seed=1) for _ in prompts]
    got = eng.generate(prompts, sp)

    torch.manual_seed(4)
    model2 = DeepseekV2ForCausalLM(cfg, device=device).to(device, torch.float32)
    plain_cfg = EngineConfig(
        model=cfg, cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16),
    )
    plain_cfg.validate()
    plain = LLMEngine(plain_cfg, device=device, dtype=torch.float32, model=model2, model_cfg=cfg)
    assert got == plain.generate(prompts, sp)