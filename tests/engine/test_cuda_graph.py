"""CUDA Graph runner 测试（集群 GPU）：回放 ≡ eager、padding 批不变、档位选择。"""

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


def _make_setup(max_batch: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    dev = "cuda"
    cfg = _tiny_cfg()
    engine_cfg = EngineConfig(
        model=cfg,
        cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(max_num_seqs=max_batch),
    )
    engine_cfg.validate()
    model = DeepseekV2ForCausalLM(cfg, device=dev).to(dev, torch.bfloat16)
    # 256 块：每 decode 行独占 7 块（见 _decode_case），12 行需 84 块
    cache = CacheEngine(cfg, engine_cfg.cache, num_blocks=256, device=dev, dtype=torch.bfloat16)
    runner = CudaGraphRunner(model, cache, engine_cfg, dev, max_batch, scratch_block=1)
    return model, cache, runner, engine_cfg


def _decode_case(n: int, seed: int = 1):
    """构造 n 个 decode 行的输入与 block 表（每 seq 独立块段）。

    位置必须落在 block 表覆盖范围内（pos < 7*bs=56）：块表短于 seqlen 时
    越界列落到保留块 0，多行写同一槽位产生竞态（此前 eager vs graph
    的 81% 不匹配即此——测试构造缺陷，非 kernel 缺陷）。
    """
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 500, (n,), generator=g).tolist()
    positions = [(i * 7) % 44 + 1 for i in range(n)]  # 1..44 < 7*8
    block_rows = []
    for i in range(n):
        row = [i * 7 + 2 + j for j in range(7)]  # 每 seq 独占 7 块（覆盖 56 token）
        block_rows.append(row)
    return tokens, positions, block_rows


def test_graph_replay_matches_eager() -> None:
    model, cache, runner, _cfg = _make_setup()
    tokens, positions, block_rows = _decode_case(n=12)
    # eager 参考（与捕获时同 num_splits ⇒ 同 kernel 序列）
    bt = torch.zeros(len(block_rows), runner.max_bt, dtype=torch.int64, device="cuda")
    for i, row in enumerate(block_rows):
        bt[i, : len(row)] = torch.tensor(row, dtype=torch.int64, device="cuda")
    hidden_eager = model.forward_hidden(
        torch.tensor(tokens, dtype=torch.int64, device="cuda"),
        torch.tensor(positions, dtype=torch.int64, device="cuda"),
        torch.arange(len(tokens), dtype=torch.int64, device="cuda"),
        [1] * len(tokens),
        bt, cache, num_splits=runner.num_splits,
    )
    hidden_graph = runner.replay(tokens, positions, block_rows)
    torch.testing.assert_close(
        hidden_graph[: len(tokens)].float(), hidden_eager.float(), rtol=1e-2, atol=1e-2
    )


def test_graph_padding_batch_invariance() -> None:
    """n=5 真实行在 16 档位回放（11 行 padding）≡ 5 行 eager：
    MoE 分桶含垃圾 padding token，行结果必须不变。"""
    model, cache, runner, _cfg = _make_setup()
    tokens, positions, block_rows = _decode_case(n=5, seed=3)
    bt = torch.zeros(5, runner.max_bt, dtype=torch.int64, device="cuda")
    for i, row in enumerate(block_rows):
        bt[i, : len(row)] = torch.tensor(row, dtype=torch.int64, device="cuda")
    hidden_eager = model.forward_hidden(
        torch.tensor(tokens, dtype=torch.int64, device="cuda"),
        torch.tensor(positions, dtype=torch.int64, device="cuda"),
        torch.arange(5, dtype=torch.int64, device="cuda"),
        [1] * 5, bt, cache, num_splits=runner.num_splits,
    )
    hidden_graph = runner.replay(tokens, positions, block_rows)  # tier 8（5→8）
    torch.testing.assert_close(
        hidden_graph[:5].float(), hidden_eager.float(), rtol=1e-2, atol=1e-2
    )


def test_graph_replay_deterministic() -> None:
    model, cache, runner, _cfg = _make_setup(seed=5)
    tokens, positions, block_rows = _decode_case(n=7, seed=6)
    a = runner.replay(tokens, positions, block_rows)
    b = runner.replay(tokens, positions, block_rows)
    assert torch.equal(a, b)


def test_graph_multiple_tiers() -> None:
    model, cache, runner, _cfg = _make_setup(max_batch=16, seed=7)
    for n in (1, 3, 9, 16):
        tokens, positions, block_rows = _decode_case(n=n, seed=100 + n)
        hidden = runner.replay(tokens, positions, block_rows)
        assert hidden.shape[0] >= n
        assert torch.isfinite(hidden[:n].float()).all()
