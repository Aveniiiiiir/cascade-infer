"""varlen + MTP + SO 集成回归（2026-11-18 审计 B1/B2/#6 的行为锁定）。

修复前的失效链（集群 xgrammar 路径曾完全无覆盖——现有 SO 测试在装了
xgrammar 的环境主动 skip，只跑从不失败的 StubGrammar）：
- B1：varlen 第一步 v=1 → K_actual=0 → masks[0] 从未填充 → 全 -inf →
  softmax NaN → _inv_cdf「全零概率行」ValueError → fail_all_inflight
  判死全部在途请求；
- B2：accept_token 返回值被无视 + 按固定 K 回滚 → fresh 行/首个草案
  非法时回滚超限抛错或静默 desync；
- #6(a)：update_mtp_accept 从不被调用 → rates 恒 0 → varlen 退化。

覆盖策略：monkeypatch HAS_XGRAMMAR=False 强制 stub 编译路径——保证
在**装有 xgrammar 的集群**上真正执行（这正是审计指出的覆盖陷阱）。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.config import CacheConfig, EngineConfig, SchedulerConfig
from cascade.engine.llm_engine import LLMEngine
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from cascade.model_executor.models.mtp_nextn import MockDraftModel
from cascade.sampling_params import SamplingParams
from cascade.structured_output.matcher import GrammarBackend, SOMatcher
from tests.engine.test_engine_mtp import _tiny_cfg


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_engine(device: str, seed: int = 0) -> LLMEngine:
    torch.manual_seed(seed)
    cfg = _tiny_cfg()
    engine_cfg = EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=256,
            nextn_draft_tokens=3, mtp_varlen=True,
        ),
    )
    engine_cfg.validate()
    model = DeepseekV2ForCausalLM(cfg, device=device).to(device, torch.float32)
    draft = MockDraftModel(cfg.vocab_size)
    return LLMEngine(
        engine_cfg, device=device, dtype=torch.float32,
        draft_model=draft, model=model, model_cfg=cfg,
    )


@pytest.mark.parametrize("device", _devices())
def test_varlen_so_mtp_end_to_end(device: str, monkeypatch) -> None:
    """varlen+MTP+SO 端到端：修复前第一步即 fail_all_inflight（B1）。

    stub 文法 "30,11,42"：贪心输出前 3 token 逐位受约束。两条并发请求
    都必须正常完成——旧行为（K_actual=0 全零掩码 → NaN）会在第一个
    decode 步判死全部在途请求。
    """
    import cascade.structured_output.compiler as compiler_mod

    monkeypatch.setattr(compiler_mod, "HAS_XGRAMMAR", False)  # 强制 stub：集群也执行
    eng = _make_engine(device)
    g = torch.Generator().manual_seed(11)
    prompts = [torch.randint(1, 500, (n,), generator=g).tolist() for n in (13, 9)]
    sp = [
        SamplingParams(
            temperature=0.0, max_tokens=10, seq_seed=i,
            so_args={"schema_type": "regex", "schema": "30,11,42"},
        )
        for i in range(len(prompts))
    ]
    out = eng.generate(prompts, sp)
    for i in range(len(prompts)):
        got = out[f"req-{i}"]
        assert got[:3] == [30, 11, 42], f"prompt {i}: SO 约束未生效 {got[:3]}"
        assert len(got) == 10, f"prompt {i}: 未完成（判死/死锁）"


@pytest.mark.parametrize("device", _devices())
def test_varlen_rates_feedback(device: str) -> None:
    """A6(a)：验证后 rates 被回灌——varlen 不再退化为 v≡1。

    varlen 开启、无 SO：跑几步后 scheduler._mtp_rates 应出现该 rid
    （真实草案行的 accept_probs 累乘；修复前生产链路从不回灌）。
    """
    eng = _make_engine(device, seed=7)
    prompt = [5, 12, 33, 7]
    sp = SamplingParams(temperature=0.0, max_tokens=12, seq_seed=0)
    eng.add_request(prompt, sp)
    seen = False
    n_steps = 0
    while eng.scheduler.has_unfinished_requests and n_steps < 100:
        eng.step()
        n_steps += 1
        if "req-0" in eng.scheduler._mtp_rates:
            seen = True
            break
    assert seen, "rates 从未被回灌——varlen 仍退化为 v≡1（审计 #6(a)）"
    while eng.scheduler.has_unfinished_requests:  # 收尾
        eng.step()


def test_rejecting_backend_protocol() -> None:
    """B2 协议锁定：accept_token 必须返回 bool（False = 非法且状态未推进），
    SOMatcher 透传；_verify_mtp_seq 的试探循环据此计回滚数。"""

    class RejectingBackend(GrammarBackend):
        """只接受 token 7；其余返回 False（模拟 xgrammar 拒绝）。"""

        def __init__(self) -> None:
            self.rolled_back = 0

        def fill_mask_into(self, words, vocab_size: int) -> None:
            for i in range(len(words)):
                words[i] = -1  # 全开（掩码语义由调用方测试）
            words[7 // 32] = 1 << (7 % 32)

        def accept_token(self, token_id: int) -> bool:
            return token_id == 7

        def rollback(self, k: int) -> None:
            self.rolled_back = k

        @property
        def terminated(self) -> bool:
            return False

    backend = RejectingBackend()
    m = SOMatcher(backend, vocab_size=512)
    assert m.accept_token(7) is True, "合法 token 应返回 True"
    assert m.accept_token(8) is False, "非法 token 应返回 False（状态未推进）"
    m.rollback(0)
    assert backend.rolled_back == 0, "rollback(0) 应为 no-op"
    m.rollback(2)
    assert backend.rolled_back == 2
