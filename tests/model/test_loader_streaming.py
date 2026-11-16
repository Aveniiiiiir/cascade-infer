"""加载器内存行为测试（torch + safetensors）：
- 按目标 dtype 直接构造骨架（fp32 骨架 = 15.7B×4B ≈ 63GB，62GB 机器必 OOM）
- 多 shard 流式赋值（峰值 = 骨架 + 单 shard，而非骨架 + 全量 state）
- missing 报告按 (堆叠参数, 专家) 粒度
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from cascade.config import ModelConfig
from cascade.model_executor.model_loader import _assign, _construct_model, load_deepseek_v2


def _tiny_cfg_dict() -> dict:
    return {
        "model_type": "deepseek_v2",
        "architectures": ["DeepseekV2ForCausalLM"],
        "vocab_size": 128,
        "max_position_embeddings": 64,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 8,
        "v_head_dim": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 16,
        "first_k_dense_replace": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }


def test_construct_model_uses_target_dtype() -> None:
    cfg = ModelConfig.from_hf_dict(_tiny_cfg_dict())
    m = _construct_model(cfg, torch.bfloat16)
    dtypes = {p.dtype for p in m.parameters()}
    assert dtypes == {torch.bfloat16}, dtypes


def test_multishard_streaming_roundtrip(tmp_path: Path) -> None:
    cfg_dict = _tiny_cfg_dict()
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")
    cfg = ModelConfig.from_hf_dict(cfg_dict)
    src = _construct_model(cfg, torch.bfloat16)
    sd = {k: v.detach().contiguous() for k, v in src.state_dict().items()}
    keys = sorted(sd)
    half = len(keys) // 2
    from safetensors.torch import save_file

    weight_map: dict[str, str] = {}
    for i, ks in enumerate((keys[:half], keys[half:])):
        name = f"model-0000{i + 1}-of-00002.safetensors"
        save_file({k: sd[k] for k in ks}, str(tmp_path / name))
        weight_map.update({k: name for k in ks})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    model, loaded_cfg = load_deepseek_v2(tmp_path, device="cpu", dtype=torch.bfloat16)
    assert loaded_cfg.hidden_size == 32
    out = model.state_dict()
    for k in keys:
        assert torch.equal(out[k], sd[k]), k
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


def test_missing_report_flags_individual_experts() -> None:
    cfg = ModelConfig.from_hf_dict(_tiny_cfg_dict())
    model = _construct_model(cfg, torch.bfloat16)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # state_dict 存的是堆叠张量（w1/w3/w2）；HF checkpoint 用逐专家键
    # （mlp.experts.E.*.weight）——删除其一以验证按专家粒度的 missing 上报
    hf_expert_key = "model.layers.1.mlp.experts.2.gate_proj.weight"
    assert "model.layers.1.mlp.w1" in sd
    sd[hf_expert_key] = sd.pop("model.layers.1.mlp.w1")[2].clone()
    rep = _assign(model, sd, torch.bfloat16)
    # 只提供 expert 2 的 gate_proj（其余 0/1/3 缺失）→ 必须逐片上报，
    # 而不是整层静默通过
    assert set(rep.missing) == {
        "model.layers.1.mlp.w1[0]",
        "model.layers.1.mlp.w1[1]",
        "model.layers.1.mlp.w1[3]",
    }, rep.missing
    assert not rep.unexpected


def test_non_tied_checkpoint_missing_lm_head_is_reported() -> None:
    cfg = ModelConfig.from_hf_dict(_tiny_cfg_dict())
    model = _construct_model(cfg, torch.bfloat16)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sd.pop("lm_head.weight")
    rep = _assign(model, sd, torch.bfloat16, tie_word_embeddings=False)
    assert "lm_head.weight" in rep.missing
    # tie 模型则不应报（共享 embedding 权重）
    rep_tied = _assign(model, sd, torch.bfloat16, tie_word_embeddings=True)
    assert "lm_head.weight" not in rep_tied.missing
