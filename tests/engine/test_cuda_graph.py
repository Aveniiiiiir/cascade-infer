"""CUDA Graph runner 测试（集群 GPU）：回放 ≡ eager、padding 批不变、档位选择。

覆盖两条路径：
- v=1 普通 decode（segment_lens 全 1 → split-K decode kernel）
- v=1+K 投机验证批（uniform-varlen prefill kernel；MTP / n-gram 用）
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA graph 测试需要 CUDA", allow_module_level=True)

from cascade.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from cascade.worker.cache_engine import CacheEngine
from cascade.worker.cuda_graph_runner import CudaGraphRunner


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        model_type="deepseek_v2",
        max_model_len=256,
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


def _make_setup(max_batch: int = 16, seed: int = 0, k: int = 0, kv_dtype: str = "bf16"):
    torch.manual_seed(seed)
    dev = "cuda"
    cfg = _tiny_cfg()
    engine_cfg = EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=8, kv_dtype=kv_dtype),
        scheduler=SchedulerConfig(max_num_seqs=max_batch, nextn_draft_tokens=k),
    )
    engine_cfg.validate()
    model = DeepseekV2ForCausalLM(cfg, device=dev).to(dev, torch.bfloat16)
    # 256 块：每 decode seq 独占 7 块（见 _decode_case），16 seq 需 112 块
    cache = CacheEngine(cfg, engine_cfg.cache, num_blocks=256, device=dev, dtype=torch.bfloat16)
    runner = CudaGraphRunner(model, cache, engine_cfg, dev, max_batch, scratch_block=1)
    return model, cache, runner, engine_cfg


def _decode_case(n: int, seed: int = 1, v: int = 1):
    """构造 n 个 decode seq 的逐行输入与 block 表（每 seq 独立块段）。

    位置必须落在 block 表覆盖范围内（pos+v-1 < 7*bs=56）：块表短于 seqlen 时
    越界列落到保留块 0，多行写同一槽位产生竞态（此前 eager vs graph
    的 81% 不匹配即此——测试构造缺陷，非 kernel 缺陷）。
    """
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 500, (n, v), generator=g).tolist()
    positions = [[(i * 7 + j) % (56 - v) for j in range(v)] for i in range(n)]
    block_rows = []
    for i in range(n):
        row = [i * 7 + 2 + j for j in range(7)]  # 每 seq 独占 7 块（覆盖 56 token）
        block_rows.append(row)
    return tokens, positions, block_rows


def _eager_forward(model, cache, runner, tokens_rows, positions_rows, block_rows):
    v = len(tokens_rows[0])
    flat_tokens = [t for row in tokens_rows for t in row]
    flat_positions = [p for row in positions_rows for p in row]
    n = len(tokens_rows)
    bt = torch.zeros(n, runner.max_bt, dtype=torch.int64, device="cuda")
    for i, row in enumerate(block_rows):
        bt[i, : len(row)] = torch.tensor(row, dtype=torch.int64, device="cuda")
    return model.forward_hidden(
        torch.tensor(flat_tokens, dtype=torch.int64, device="cuda"),
        torch.tensor(flat_positions, dtype=torch.int64, device="cuda"),
        torch.arange(n, dtype=torch.int64, device="cuda").repeat_interleave(v),
        [v] * n,
        bt, cache, num_splits=runner.num_splits,
    )


def test_graph_replay_matches_eager() -> None:
    model, cache, runner, _cfg = _make_setup()
    tokens_rows, positions_rows, block_rows = _decode_case(n=12)
    # eager 参考（与捕获时同 num_splits ⇒ 同 kernel 序列）
    hidden_eager = _eager_forward(model, cache, runner, tokens_rows, positions_rows, block_rows)
    hidden_graph = runner.replay(tokens_rows, positions_rows, block_rows)
    torch.testing.assert_close(
        hidden_graph[: len(tokens_rows)].float(), hidden_eager.float(), rtol=1e-2, atol=1e-2
    )


def test_graph_padding_batch_invariance() -> None:
    """n=5 真实 seq 在 16 档位回放（11 个 padding seq）≡ 5 seq eager：
    MoE 分桶含垃圾 padding token，行结果必须不变。"""
    model, cache, runner, _cfg = _make_setup()
    tokens_rows, positions_rows, block_rows = _decode_case(n=5, seed=3)
    hidden_eager = _eager_forward(model, cache, runner, tokens_rows, positions_rows, block_rows)
    hidden_graph = runner.replay(tokens_rows, positions_rows, block_rows)  # tier 8（5→8）
    torch.testing.assert_close(
        hidden_graph[:5].float(), hidden_eager.float(), rtol=1e-2, atol=1e-2
    )


def test_graph_replay_deterministic() -> None:
    model, cache, runner, _cfg = _make_setup(seed=5)
    tokens_rows, positions_rows, block_rows = _decode_case(n=7, seed=6)
    a = runner.replay(tokens_rows, positions_rows, block_rows)
    b = runner.replay(tokens_rows, positions_rows, block_rows)
    assert torch.equal(a, b)


def test_graph_multiple_tiers() -> None:
    model, cache, runner, _cfg = _make_setup(max_batch=16, seed=7)
    for n in (1, 3, 9, 16):
        tokens_rows, positions_rows, block_rows = _decode_case(n=n, seed=100 + n)
        hidden = runner.replay(tokens_rows, positions_rows, block_rows)
        assert hidden.shape[0] >= n
        assert torch.isfinite(hidden[:n].float()).all()


def test_graph_multi_token_replay_matches_eager() -> None:
    """投机验证批（每 seq 4 行）图回放 ≡ eager（uniform-varlen 路径）。"""
    model, cache, runner, _cfg = _make_setup(k=3)
    assert runner.tokens_per_seq == 4
    tokens_rows, positions_rows, block_rows = _decode_case(n=6, seed=9, v=4)
    hidden_eager = _eager_forward(model, cache, runner, tokens_rows, positions_rows, block_rows)
    hidden_graph = runner.replay(tokens_rows, positions_rows, block_rows)
    torch.testing.assert_close(
        hidden_graph[: 6 * 4].float(), hidden_eager.float(), rtol=1e-2, atol=1e-2
    )


def test_graph_multi_token_padding_invariance() -> None:
    model, cache, runner, _cfg = _make_setup(k=2)
    assert runner.tokens_per_seq == 3
    tokens_rows, positions_rows, block_rows = _decode_case(n=3, seed=13, v=3)
    hidden_eager = _eager_forward(model, cache, runner, tokens_rows, positions_rows, block_rows)
    hidden_graph = runner.replay(tokens_rows, positions_rows, block_rows)  # tier 4
    torch.testing.assert_close(
        hidden_graph[: 3 * 3].float(), hidden_eager.float(), rtol=1e-2, atol=1e-2
    )


def test_graph_fp8_replay_matches_eager() -> None:
    """FP8 KV 池（量化+dequant 全部在图内）：回放 ≡ eager。"""
    model, cache, runner, _cfg = _make_setup(kv_dtype="fp8", seed=11)
    assert cache.kv_dtype == "fp8"
    tokens_rows, positions_rows, block_rows = _decode_case(n=6, seed=12)
    hidden_eager = _eager_forward(model, cache, runner, tokens_rows, positions_rows, block_rows)
    hidden_graph = runner.replay(tokens_rows, positions_rows, block_rows)
    torch.testing.assert_close(
        hidden_graph[: len(tokens_rows)].float(), hidden_eager.float(), rtol=1e-2, atol=1e-2
    )


def test_graph_fp8_multi_token_replay() -> None:
    model, cache, runner, _cfg = _make_setup(k=3, kv_dtype="fp8", seed=21)
    tokens_rows, positions_rows, block_rows = _decode_case(n=4, seed=22, v=4)
    hidden_eager = _eager_forward(model, cache, runner, tokens_rows, positions_rows, block_rows)
    hidden_graph = runner.replay(tokens_rows, positions_rows, block_rows)
    torch.testing.assert_close(
        hidden_graph[: 4 * 4].float(), hidden_eager.float(), rtol=1e-2, atol=1e-2
    )