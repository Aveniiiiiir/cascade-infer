"""ModelConfig HF 解析测试（纯 stdlib，Windows 可跑）。

fixture 采用 DeepSeek-V2-Lite 公开 config 的字段结构（值以解析契约为准）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade.config import ModelConfig

FIXTURES = Path(__file__).parent / "fixtures"

V2_LITE_CONFIG = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "vocab_size": 102400,
    "max_position_embeddings": 4096,
    "hidden_size": 2048,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "num_key_value_heads": 1,
    "q_lora_rank": 1024,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "intermediate_size": 2048,
    "moe_intermediate_size": 1408,
    "first_k_dense_replace": 1,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "num_experts_per_tok": 6,
    "routed_scaling_factor": 1.0,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "norm_topk_prob": False,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000,
    "tie_word_embeddings": False,
}


def test_from_hf_dict_v2_lite() -> None:
    cfg = ModelConfig.from_hf_dict(V2_LITE_CONFIG)
    assert cfg.num_layers == 27
    assert cfg.vocab_size == 102400
    assert cfg.max_model_len == 4096
    assert cfg.num_attention_heads == 16
    assert cfg.num_experts_per_tok == 6
    assert cfg.has_e_score_correction_bias  # noaux_tc
    # 派生量
    assert cfg.kv_cache_dim == 512 + 64
    assert cfg.qk_head_dim == 128 + 64
    assert abs(cfg.softmax_scale - (192**-0.5)) < 1e-12
    assert cfg.num_dense_layers == 1
    assert cfg.num_moe_layers == 26
    assert cfg.shared_intermediate_size == 2 * 1408
    assert not cfg.has_mtp


def test_from_hf_dict_defaults_for_v2() -> None:
    """V2 236B 风格：softmax 评分 / greedy topk 无 bias。"""
    d = dict(V2_LITE_CONFIG)
    d["scoring_func"] = "softmax"
    d["topk_method"] = "greedy"
    cfg = ModelConfig.from_hf_dict(d)
    assert not cfg.has_e_score_correction_bias
    assert cfg.scoring_func == "softmax"


def test_from_hf_dict_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unsupported model_type"):
        ModelConfig.from_hf_dict({"model_type": "llama", "vocab_size": 1, "num_hidden_layers": 1,
                                  "hidden_size": 1, "num_attention_heads": 1, "kv_lora_rank": 1,
                                  "qk_rope_head_dim": 2, "qk_nope_head_dim": 2, "v_head_dim": 2})


def test_from_hf_dict_rejects_v3_explicitly() -> None:
    """审查 #8：V3 的 n_group 组路由 / rope beta / nextn 权重未实现——
    显式拒绝，绝不静默半支持（错误路由 = 静默错误输出）。"""
    d = dict(V2_LITE_CONFIG)
    d["model_type"] = "deepseek_v3"
    with pytest.raises(NotImplementedError, match="deepseek_v3"):
        ModelConfig.from_hf_dict(d)


def test_mla_kv_heads_normalized_to_single_stream() -> None:
    """真实 V2-Lite config 的 num_key_value_heads=16（写成注意力头数），
    MLA 的 KV 经 kv_a_proj_with_mqa 压缩为单流 latent —— 归一化为 1。"""
    for n in (8, 16):
        d = dict(V2_LITE_CONFIG)
        d["num_key_value_heads"] = n
        cfg = ModelConfig.from_hf_dict(d)
        assert cfg.num_kv_heads == 1


def test_fixture_file_roundtrip(tmp_path: Path) -> None:
    """config.json 文件 → from_hf_dict（集群路径的真实用法）。"""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(V2_LITE_CONFIG), encoding="utf-8")
    cfg = ModelConfig.from_hf_dict(json.loads(p.read_text(encoding="utf-8")))
    assert cfg.hidden_size == 2048
