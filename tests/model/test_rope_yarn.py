"""YaRN rope 单元测试：与 HF `_compute_yarn_parameters` 公式逐项对拍。

V2-Lite 的真实 rope_scaling：
  {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096,
   "beta_fast": 32, "beta_slow": 1, "mscale": 0.707, "mscale_all_dim": 0.707}
mscale == mscale_all_dim ⇒ attention factor = 1.0（幅度不缩放）。
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cascade.model_executor.layers.rotary import RotaryEmbedding, yarn_get_mscale

HEAD_DIM = 64
THETA = 10000.0

V2_LITE_SCALING = {
    "type": "yarn",
    "factor": 40,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
}


def _ref_yarn_inv_freq(head_dim=HEAD_DIM, theta=THETA, rs=V2_LITE_SCALING):
    """HF transformers `_compute_yarn_parameters` 的独立复刻。"""
    factor = float(rs["factor"])
    orig = int(rs["original_max_position_embeddings"])
    beta_fast = float(rs["beta_fast"])
    beta_slow = float(rs["beta_slow"])

    def corr_dim(num_rotations):
        return (head_dim * math.log(orig / (num_rotations * 2 * math.pi))) / (
            2 * math.log(theta)
        )

    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), head_dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp(
        (torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
    )
    pos_freqs = theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)
    mask = 1.0 - ramp
    return inv_interp * (1.0 - mask) + inv_extrap * mask


def test_yarn_inv_freq_matches_hf_formula() -> None:
    rope = RotaryEmbedding(HEAD_DIM, THETA, device="cpu", rope_scaling=V2_LITE_SCALING)
    ref = _ref_yarn_inv_freq()
    torch.testing.assert_close(rope.inv_freq.float(), ref, rtol=1e-6, atol=1e-9)


def test_yarn_attention_factor_is_one_when_mscale_equal() -> None:
    rope = RotaryEmbedding(HEAD_DIM, THETA, device="cpu", rope_scaling=V2_LITE_SCALING)
    assert abs(rope.attention_factor - 1.0) < 1e-12
    # 若二者不同，attention factor 应等于比值
    rs = dict(V2_LITE_SCALING, mscale=1.0, mscale_all_dim=0.5)
    rope2 = RotaryEmbedding(HEAD_DIM, THETA, device="cpu", rope_scaling=rs)
    expected = yarn_get_mscale(40, 1.0) / yarn_get_mscale(40, 0.5)
    assert abs(rope2.attention_factor - expected) < 1e-12


def test_yarn_cos_sin_matches_reference() -> None:
    rope = RotaryEmbedding(HEAD_DIM, THETA, device="cpu", rope_scaling=V2_LITE_SCALING)
    pos = torch.arange(64, dtype=torch.float32)
    freqs = torch.outer(pos, _ref_yarn_inv_freq())
    emb = torch.cat((freqs, freqs), dim=-1)
    ref_cos, ref_sin = emb.cos(), emb.sin()
    got_cos, got_sin = rope.cos_sin(64)
    torch.testing.assert_close(got_cos.float(), ref_cos, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_sin.float(), ref_sin, rtol=1e-6, atol=1e-6)


def test_rope_without_scaling_is_plain() -> None:
    rope = RotaryEmbedding(HEAD_DIM, THETA, device="cpu")
    ref = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    torch.testing.assert_close(rope.inv_freq.float(), ref, rtol=1e-6, atol=1e-12)
    assert rope.attention_factor == 1.0


def test_apply_shapes_and_dtype() -> None:
    rope = RotaryEmbedding(HEAD_DIM, THETA, device="cpu")
    rope.to("cpu", torch.bfloat16)
    T, H = 8, 4
    q = torch.randn(T, H, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(T, HEAD_DIM, dtype=torch.bfloat16)
    pos = torch.arange(T)
    q2, k2 = rope.apply(q, k, pos)
    assert q2.shape == q.shape and k2.shape == k.shape
    assert q2.dtype == torch.bfloat16 and k2.dtype == torch.bfloat16
    # 位置 0：cos=1, sin=0 → q 不变
    torch.testing.assert_close(q2[0], q[0])
