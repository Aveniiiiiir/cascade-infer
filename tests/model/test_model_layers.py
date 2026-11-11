"""模型层正确性测试（需要 torch；CUDA 可用则用 GPU，否则 CPU）。

四条核心不变式（全部用 fp32 + 小随机权重，容差收紧）：
1. MLA 显式路径 ≡ 吸收路径（数学等价）
2. 增量 decode（teacher forcing）≡ 整段 prefill（KV store/gather/rope 正确性）
3. chunked prefill ≡ 整段 prefill
4. 多段混合批 ≡ 各 seq 单独跑（模型级批不变）
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.config import CacheConfig, ModelConfig
from cascade.model_executor.layers.mla import MLAAttention
from cascade.model_executor.layers.moe import DeepseekMoE
from cascade.model_executor.layers.rotary import RotaryEmbedding
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from cascade.worker.cache_engine import CacheEngine


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def tiny_cfg(**overrides) -> ModelConfig:
    defaults = dict(
        model_type="deepseek_v2",
        max_model_len=256,
        vocab_size=256,
        num_layers=2,
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
    defaults.update(overrides)
    return ModelConfig(**defaults)


def make_cache(cfg: ModelConfig, num_blocks: int = 64, block_size: int = 8) -> CacheEngine:
    return CacheEngine(cfg, CacheConfig(block_size=block_size), num_blocks, device=_device(), dtype=torch.float32)


def rand_tokens(n: int, vocab: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g)


def one_seq_block_table(num_tokens: int, block_size: int, num_blocks: int) -> torch.Tensor:
    """单 seq：按序占用 block，预分配 ceil(num_tokens/bs)+1 个。"""
    need = (num_tokens + block_size - 1) // block_size + 1
    assert need <= num_blocks
    return torch.arange(1, need + 1, dtype=torch.int64, device=_device()).unsqueeze(0)


@pytest.fixture()
def seed_everything():
    torch.manual_seed(1234)
    return _device()


# --------------------------------------------------------------------- MLA


def test_mla_explicit_vs_absorbed(seed_everything):
    cfg = tiny_cfg()
    dev = _device()
    attn = MLAAttention(cfg, layer_idx=0).to(dev, torch.float32)
    cache = make_cache(cfg)
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.rope_theta, device=dev)
    S = 33
    tokens = rand_tokens(S, cfg.vocab_size, seed=7).to(dev)
    bt = one_seq_block_table(S + 4, cache.cache_cfg.block_size, cache.num_blocks)
    x = torch.randn(S, cfg.hidden_size, device=dev)
    positions = torch.arange(S, device=dev)
    seq_ids = torch.zeros(S, dtype=torch.int64, device=dev)

    out_exp = attn(x, positions, seq_ids, [S], bt, cache, rope, use_absorbed=False)
    # 重置 cache 再跑吸收路径（避免依赖写入幂等性）
    cache2 = make_cache(cfg)
    out_abs = attn(x, positions, seq_ids, [S], bt, cache2, rope, use_absorbed=True)
    torch.testing.assert_close(out_exp, out_abs, rtol=1e-4, atol=1e-4)


def test_mla_incremental_decode_equals_prefill(seed_everything):
    """teacher forcing：整段 prefill 的末位 logits ≡ 逐步 decode。"""
    cfg = tiny_cfg()
    dev = _device()
    attn = MLAAttention(cfg, layer_idx=0).to(dev, torch.float32)
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.rope_theta, device=dev)
    S = 29
    tokens = rand_tokens(S, cfg.vocab_size, seed=11).to(dev)

    # 整段 prefill（吸收路径——decode 语义的规范路径）
    cache_a = make_cache(cfg)
    bt_a = one_seq_block_table(S + 4, cache_a.cache_cfg.block_size, cache_a.num_blocks)
    x = torch.randn(S, cfg.hidden_size, device=dev)
    out_full = attn(
        x, torch.arange(S, device=dev), torch.zeros(S, dtype=torch.int64, device=dev),
        [S], bt_a, cache_a, rope, use_absorbed=True,
    )[-1]

    # 逐步 decode：先 prefill 前 K，再逐 token 喂入
    K = 10
    cache_b = make_cache(cfg)
    bt_b = one_seq_block_table(S + 4, cache_b.cache_cfg.block_size, cache_b.num_blocks)
    out = attn(
        x[:K], torch.arange(K, device=dev), torch.zeros(K, dtype=torch.int64, device=dev),
        [K], bt_b, cache_b, rope, use_absorbed=True,
    )[-1]
    for p in range(K, S):
        out = attn(
            x[p : p + 1], torch.tensor([p], device=dev),
            torch.zeros(1, dtype=torch.int64, device=dev), [1],
            bt_b, cache_b, rope, use_absorbed=True,
        )[-1]
    torch.testing.assert_close(out_full, out, rtol=1e-4, atol=1e-4)


def test_mla_chunked_equals_full(seed_everything):
    cfg = tiny_cfg()
    dev = _device()
    attn = MLAAttention(cfg, layer_idx=0).to(dev, torch.float32)
    rope = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.rope_theta, device=dev)
    S = 37
    x = torch.randn(S, cfg.hidden_size, device=dev)

    cache_a = make_cache(cfg)
    bt_a = one_seq_block_table(S + 4, cache_a.cache_cfg.block_size, cache_a.num_blocks)
    out_full = attn(
        x, torch.arange(S, device=dev), torch.zeros(S, dtype=torch.int64, device=dev),
        [S], bt_a, cache_a, rope, use_absorbed=True,
    )

    cache_b = make_cache(cfg)
    bt_b = one_seq_block_table(S + 4, cache_b.cache_cfg.block_size, cache_b.num_blocks)
    outs = []
    offset = 0
    for chunk in (15, 15, 7):  # 跨 block 边界切分（block_size=8）
        pos = torch.arange(offset, offset + chunk, device=dev)
        outs.append(
            attn(
                x[offset : offset + chunk], pos,
                torch.zeros(chunk, dtype=torch.int64, device=dev), [chunk],
                bt_b, cache_b, rope, use_absorbed=True,
            )
        )
        offset += chunk
    out_chunked = torch.cat(outs, dim=0)
    torch.testing.assert_close(out_full, out_chunked, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------- MoE


def test_moe_gate_routing_oracle(seed_everything):
    cfg = tiny_cfg(n_routed_experts=8, num_experts_per_tok=3)
    dev = _device()
    moe = DeepseekMoE(cfg).to(dev, torch.float32)
    T = 64
    x = torch.randn(T, cfg.hidden_size, device=dev)
    idx, w = moe.gate(x)

    # 手工 oracle
    logits = x @ moe.gate.weight.T
    scores = logits.sigmoid()
    biased = scores + moe.gate.e_score_correction_bias.unsqueeze(0)
    ref_idx = torch.topk(biased, k=3, dim=-1, sorted=False)[1]
    ref_w = scores.gather(1, ref_idx) * cfg.routed_scaling_factor
    # topk 无 sorted → 排序无关比对
    assert _setwise_equal(idx, ref_idx), "路由选择与 oracle 不一致"
    torch.testing.assert_close(w, ref_w, rtol=1e-5, atol=1e-6)


def _setwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return all(set(ai.tolist()) == set(bi.tolist()) for ai, bi in zip(a, b))


def test_moe_output_oracle(seed_everything):
    cfg = tiny_cfg(n_routed_experts=8, num_experts_per_tok=3, routed_scaling_factor=2.5, norm_topk_prob=True)
    dev = _device()
    moe = DeepseekMoE(cfg).to(dev, torch.float32)
    T = 32
    x = torch.randn(T, cfg.hidden_size, device=dev)
    out = moe(x)

    idx, w = moe.gate(x)
    ref = torch.zeros_like(x)
    for t in range(T):
        for k in range(cfg.num_experts_per_tok):
            e = idx[t, k].item()
            ref[t] += w[t, k] * moe.expert_ffn_ref(x[t : t + 1], e)[0]
    ref = ref + moe.shared_experts(x)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------- whole model


def _run_model(
    model: DeepseekV2ForCausalLM,
    cache: CacheEngine,
    tokens: list[torch.Tensor],  # 每段一个 tensor
    start_positions: list[int],
) -> torch.Tensor:
    """按段跑一次 forward → 拼接 logits [T, vocab]。

    block 表简化：每段（seq 行）从块 1 开始顺序分配 ceil(段末位置/bs)+1 个。
    """
    dev = cache.device
    bs = cache.cache_cfg.block_size
    seq_rows = len(tokens)
    segment_lens = [int(t.numel()) for t in tokens]
    flat = torch.cat(tokens)
    positions = torch.cat(
        [
            torch.arange(start_positions[i], start_positions[i] + segment_lens[i], device=dev)
            for i in range(seq_rows)
        ]
    )
    seq_ids = torch.cat(
        [torch.full((segment_lens[i],), i, dtype=torch.int64, device=dev) for i in range(seq_rows)]
    )
    need = [
        (start_positions[i] + segment_lens[i] + bs - 1) // bs + 1 for i in range(seq_rows)
    ]
    bt = torch.zeros(seq_rows, max(need) + 1, dtype=torch.int64, device=dev)
    for i in range(seq_rows):
        bt[i, : need[i]] = torch.arange(1, need[i] + 1, device=dev)
    return model(flat, positions, seq_ids, segment_lens, bt, cache)


def test_model_batch_invariant_and_chunked(seed_everything):
    """多段混合批 ≡ 各 seq 单独跑；整段 ≡ 分块。"""
    cfg = tiny_cfg()
    dev = _device()
    model = DeepseekV2ForCausalLM(cfg, device=dev).to(dev, torch.float32)
    tokens_a = rand_tokens(21, cfg.vocab_size, seed=1).to(dev)
    tokens_b = rand_tokens(13, cfg.vocab_size, seed=2).to(dev)

    # 混合批
    cache_m = make_cache(cfg, num_blocks=128)
    logits_mixed = _run_model(model, cache_m, [tokens_a, tokens_b], [0, 0])
    # 单独跑
    cache_a = make_cache(cfg, num_blocks=128)
    logits_a = _run_model(model, cache_a, [tokens_a], [0])
    cache_b = make_cache(cfg, num_blocks=128)
    logits_b = _run_model(model, cache_b, [tokens_b], [0])
    torch.testing.assert_close(logits_mixed[: 21], logits_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(logits_mixed[21:], logits_b, rtol=1e-4, atol=1e-4)

    # chunked prefill（A 分 3 块 + B 解码态 1 段）
    cache_c = make_cache(cfg, num_blocks=128)
    logits_c = _run_model(model, cache_c, [tokens_a[:8], tokens_a[8:16], tokens_a[16:], tokens_b], [0, 8, 16, 0])
    torch.testing.assert_close(logits_c[: 21], logits_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(logits_c[21:], logits_b, rtol=1e-4, atol=1e-4)


def test_model_incremental_decode_equals_prefill(seed_everything):
    """端到端：teacher forcing 下逐步 decode 的 logits ≡ 整段 prefill。"""
    cfg = tiny_cfg()
    dev = _device()
    model = DeepseekV2ForCausalLM(cfg, device=dev).to(dev, torch.float32)
    S = 17
    tokens = rand_tokens(S, cfg.vocab_size, seed=5).to(dev)

    cache_full = make_cache(cfg, num_blocks=128)
    logits_full = _run_model(model, cache_full, [tokens], [0])

    # prefill 前 6 个，再逐 token
    cache_inc = make_cache(cfg, num_blocks=128)
    K = 6
    logits = _run_model(model, cache_inc, [tokens[:K]], [0])
    rows = [logits[-1]]
    for p in range(K, S):
        logits = _run_model(model, cache_inc, [tokens[p : p + 1]], [p])
        rows.append(logits[-1])
    inc = torch.stack(rows)  # 位置 K-1..S-1 的 logits
    torch.testing.assert_close(inc, logits_full[K - 1 :], rtol=1e-4, atol=1e-4)
