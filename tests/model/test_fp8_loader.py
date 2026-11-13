"""FP8 checkpoint 加载测试（torch + safetensors；合成迷你 V3 风格权重）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from cascade.model_executor.model_loader import (
    _dequant_blockwise,
    _dequant_fp8_state,
    load_deepseek_v2,
)


def test_dequant_blockwise_math() -> None:
    torch.manual_seed(0)
    w = (torch.randn(128, 128) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), 0.25, dtype=torch.float32)
    out = _dequant_blockwise(w, scale)
    ref = (w.to(torch.float32) * 0.25).to(torch.bfloat16)
    assert torch.equal(out, ref)


def test_dequant_fp8_state_passthrough_and_scale() -> None:
    torch.manual_seed(1)
    w = torch.randn(256, 128).to(torch.float8_e4m3fn)
    norm = torch.ones(128)
    scale = torch.full((2, 1), 2.0)
    state = {"a.weight": w, "a.weight_scale_inv": scale, "a.norm.weight": norm}
    out = _dequant_fp8_state(state)
    assert "a.weight_scale_inv" not in out
    assert torch.equal(out["a.norm.weight"], norm)
    assert out["a.weight"].dtype == torch.bfloat16
    ref = (w.to(torch.float32) * 2.0).to(torch.bfloat16)
    assert torch.equal(out["a.weight"], ref)


def test_load_fp8_checkpoint(tmp_path: Path) -> None:
    """迷你 V3 风格 checkpoint（fp8 + 128 blockwise 尺度）→ 加载反量化。"""
    H = 128
    cfg = {
        "model_type": "deepseek_v2",
        "architectures": ["DeepseekV2ForCausalLM"],
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"},
        "vocab_size": 512,
        "max_position_embeddings": 512,
        "hidden_size": H,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "intermediate_size": 128,
        "moe_intermediate_size": 128,
        "first_k_dense_replace": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 2,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-5,
    }
    # q_a_proj: [q_lora_rank=32, hidden_size=128]（输出维是 rank 不是 H）
    q_lora = int(cfg["q_lora_rank"])
    w = torch.randn(q_lora, H).to(torch.float8_e4m3fn)
    # 128 blockwise 尺度：[ceil(rank/128), ceil(H/128)] = [1, 1]
    scale = torch.full(((q_lora + 127) // 128, (H + 127) // 128), 0.5, dtype=torch.float32)
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    from safetensors.torch import save_file

    save_file(
        {
            "model.layers.0.self_attn.q_a_proj.weight": w,
            "model.layers.0.self_attn.q_a_proj.weight_scale_inv": scale,
        },
        str(tmp_path / "model.safetensors"),
    )
    model, loaded_cfg = load_deepseek_v2(tmp_path, device="cpu", dtype=torch.bfloat16)
    got = model.model.layers[0].self_attn.q_a_proj.weight
    ref = (w.to(torch.float32) * 0.5).to(torch.bfloat16)
    assert torch.equal(got, ref)
    assert loaded_cfg.hidden_size == H
