"""RoPE（NeoX 风格 rotate_half，DeepSeek V2 家族语义）。

支持 YaRN 长度外推（V2-Lite/V2.5 的 `rope_scaling.type == "yarn"`）：
- inv_freq 按「插值(低 freq) ↔ 外推(高 freq)」的线性 ramp 混合
  （与 HF transformers `_compute_yarn_parameters` / vLLM / DeepInfer
  `YaRNScalingRotaryEmbedding` 同一公式）
- cos/sin 乘 attention factor = mscale(scale, mscale) / mscale(scale,
  mscale_all_dim)（V2-Lite 两者相等 → 1.0）

cos/sin 缓冲按需增长（位置可能超过 max_position_embeddings 的投机余量）。
"""

from __future__ import annotations

import math

import torch

# --------------------------------------------------------------------- YaRN 数学

def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype, device: str | torch.device
) -> torch.Tensor:
    if low == high:  # 防除零
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype, device=device) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class RotaryEmbedding:
    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        device: str = "cuda",
        rope_scaling: dict | None = None,
    ) -> None:
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.theta = theta
        self.device = device
        self.rope_scaling = rope_scaling
        self.dtype = torch.float32
        if rope_scaling and rope_scaling.get("type") == "yarn":
            self.attention_factor = self._yarn_attention_factor()
            inv_freq = self._yarn_inv_freq()
        else:
            self.attention_factor = 1.0
            inv_freq = 1.0 / (
                theta
                ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
            )
        self.inv_freq = inv_freq  # [dim/2]
        self._max_cached = 0
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._ensure(1024)

    # ------------------------------------------------------------------ YaRN 参数

    def _yarn_attention_factor(self) -> float:
        rs = self.rope_scaling
        scale = float(rs.get("factor", 1.0))
        mscale = rs.get("mscale")
        mscale_all_dim = rs.get("mscale_all_dim")
        if mscale is not None and mscale_all_dim is not None:
            return yarn_get_mscale(scale, float(mscale)) / yarn_get_mscale(
                scale, float(mscale_all_dim)
            )
        return 1.0

    def _yarn_inv_freq(self) -> torch.Tensor:
        rs = self.rope_scaling
        factor = float(rs.get("factor", 1.0))
        original_max = int(rs.get("original_max_position_embeddings", 4096))
        beta_fast = float(rs.get("beta_fast", 32))
        beta_slow = float(rs.get("beta_slow", 1))
        extrapolation_factor = float(rs.get("extrapolation_factor", 1.0))
        pos_freqs = self.theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device)
            / self.head_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, self.head_dim, self.theta, original_max
        )
        inv_freq_mask = (
            1.0
            - _yarn_linear_ramp_mask(
                low, high, self.head_dim // 2, torch.float32, self.device
            )
        ) * extrapolation_factor
        return inv_freq_interpolation * (1.0 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    # ------------------------------------------------------------------ 设备/精度

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> RotaryEmbedding:
        """把 cos/sin/inv_freq 缓冲迁到目标设备（可选 dtype）。

        真实模型路径：`DeepseekV2ForCausalLM(cfg, device="cpu")` 构造后
        `model.to(cuda)`——RotaryEmbedding 不是 nn.Module，其缓冲不随
        model.to 迁移，必须显式调用（e2e 真实模型首次上卡暴露）。

        dtype 与模型一致（HF 的 rope 在模型 dtype 下计算），避免每次
        apply 都做 fp32→bf16 的整表 cast。
        """
        self.device = str(device)
        self.inv_freq = self.inv_freq.to(device)
        if dtype is not None and dtype != self.dtype:
            self.dtype = dtype
            self._max_cached = 0  # 触发按新 dtype 重建
            self._ensure(1024)
        if self._cos is not None:
            self._cos = self._cos.to(device)
        if self._sin is not None:
            self._sin = self._sin.to(device)
        return self

    def _ensure(self, max_pos: int) -> None:
        if max_pos <= self._max_cached:
            return
        # 留 2 倍余量避免频繁重建
        new_max = max(max_pos * 2, 1024)
        pos = torch.arange(new_max, dtype=torch.float32, device=self.device)
        freqs = torch.outer(pos, self.inv_freq.to(torch.float32))  # [P, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [P, dim]
        cos = emb.cos() * self.attention_factor
        sin = emb.sin() * self.attention_factor
        self._cos = cos.to(self.dtype)
        self._sin = sin.to(self.dtype)
        self._max_cached = new_max

    def cos_sin(self, max_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """前 max_pos 个位置的 (cos, sin)（测试/参考实现用）。"""
        self._ensure(max_pos)
        return self._cos[:max_pos], self._sin[:max_pos]

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        ensure: bool = True,
        interleaved: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q/k: [..., head_dim]（最后一维 rope）。positions: 同 broadcast 到 [...,] 的绝对位置。

        NeoX rotate_half: x' = x*cos + rotate_half(x)*sin，
        rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])。

        interleaved=True（DeepSeek 系默认）：输入 q_pe/k_pe 为 interleaved
        布局（实部/虚部成对），先转 half-split 再旋转——与官方
        `apply_rotary_pos_emb` 的 q.view(d//2, 2).transpose().reshape() 一致。
        此前按 half-split 直接处理导致整个 rope 语义错误（位置 0 都不恒等，
        真实模型 e2e 20/20 分歧的根因）。

        ensure=True（默认，测试/直调安全）：按需扩容缓冲（含一次
        positions.max().item() 的 CPU-GPU 同步）。
        ensure=False（引擎热路径/图捕获）：**跳过检查与同步**——调用方必须
        保证容量已由 `_ensure(runtime_max_model_len)` 预热（审查 #5/#20：
        捕获区间内的 .item() 同步会让 CUDA graph 捕获必败；捕获后 _ensure
        重分配还会造成已捕获指针的 use-after-free）。
        """
        if ensure and positions.numel():
            max_pos = int(positions.max().item()) + 1
            self._ensure(max_pos)
        if interleaved:
            q = _interleaved_to_half(q)
            k = _interleaved_to_half(k)
        # cos/sin 索引后为 [T, dim]。q 为 [T, heads, dim]（多头）时需
        # unsqueeze(1) 广播到头维；k 可能是 [T, dim]（MQA 单流）或
        # [T, heads, dim]，按各自 ndim 处理（此前 q/k 共用同一 cos_p，
        # k 为 2D 时与 unsqueeze 后的 3D cos_p 维度不匹配）。
        cos_p = self._cos[positions]
        sin_p = self._sin[positions]
        if cos_p.dtype != q.dtype:  # 兜底（正常路径 dtype 已由 to() 对齐）
            cos_p = cos_p.to(q.dtype)
            sin_p = sin_p.to(q.dtype)
        cos_q = cos_p.unsqueeze(1) if q.ndim == 3 else cos_p
        sin_q = sin_p.unsqueeze(1) if q.ndim == 3 else sin_p
        cos_k = cos_p.unsqueeze(1) if k.ndim == 3 else cos_p
        sin_k = sin_p.unsqueeze(1) if k.ndim == 3 else sin_p
        q2 = q * cos_q + _rotate_half(q) * sin_q
        k2 = k * cos_k + _rotate_half(k) * sin_k
        return q2.to(q.dtype), k2.to(k.dtype)


def _interleaved_to_half(x: torch.Tensor) -> torch.Tensor:
    """interleaved → half-split（官方 DeepSeek modeling 的同一变换）。

    DeepSeek 系 checkpoint 的 q_pe/k_pe 以 interleaved 布局存储：
    每对 (d_{2i}, d_{2i+1}) 是同一频率的 (实部, 虚部)。标准 rotate_half
    需要 half-split 布局（前半实部、后半虚部）。官方
    `apply_rotary_pos_emb` 在乘 cos/sin 之前执行：
        q.view(..., d//2, 2).transpose(-1, -2).reshape(..., d)
    等价于索引重排 [0,2,4,...,1,3,5,...]。
    """
    d = x.shape[-1]
    return x.reshape(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(x.shape)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


__all__ = ["RotaryEmbedding", "_rotate_half", "yarn_get_mscale"]
