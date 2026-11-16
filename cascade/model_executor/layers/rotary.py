"""RoPE（NeoX 风格 rotate_half，DeepSeek V2 家族语义）。

V3 的 beta/mscale 变体在 `models/deepseek_v3.py` 扩展，本模块只做基础 rope。
cos/sin 缓冲按需增长（位置可能超过 max_position_embeddings 的投机余量）。
"""

from __future__ import annotations

import torch


class RotaryEmbedding:
    def __init__(self, head_dim: int, theta: float = 10000.0, device: str = "cuda") -> None:
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.theta = theta
        self.device = device
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        self.inv_freq = inv_freq  # [dim/2]
        self._max_cached = 0
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._ensure(1024)

    def to(self, device: str | torch.device) -> "RotaryEmbedding":
        """把 cos/sin/inv_freq 缓冲迁到目标设备。

        真实模型路径：`DeepseekV2ForCausalLM(cfg, device="cpu")` 构造后
        `model.to(cuda)`——RotaryEmbedding 不是 nn.Module，其缓冲不随
        model.to 迁移，必须显式调用（e2e 真实模型首次上卡暴露）。
        """
        self.device = str(device)
        self.inv_freq = self.inv_freq.to(device)
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
        freqs = torch.outer(pos, self.inv_freq)  # [P, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [P, dim]
        self._cos = emb.cos()
        self._sin = emb.sin()
        self._max_cached = new_max

    def cos_sin(self, max_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure(max_pos)
        return self._cos[:max_pos], self._sin[:max_pos]

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        ensure: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q/k: [..., head_dim]（最后一维 rope）。positions: 同 broadcast 到 [...,] 的绝对位置。

        NeoX rotate_half: x' = x*cos + rotate_half(x)*sin，
        rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])。

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
        # cos/sin 索引后为 [T, dim]。q 为 [T, heads, dim]（多头）时需
        # unsqueeze(1) 广播到头维；k 可能是 [T, dim]（MQA 单流）或
        # [T, heads, dim]，按各自 ndim 处理（此前 q/k 共用同一 cos_p，
        # k 为 2D 时与 unsqueeze 后的 3D cos_p 维度不匹配）。
        cos_p = self._cos[positions]
        sin_p = self._sin[positions]
        cos_q = cos_p.unsqueeze(1) if q.ndim == 3 else cos_p
        sin_q = sin_p.unsqueeze(1) if q.ndim == 3 else sin_p
        cos_k = cos_p.unsqueeze(1) if k.ndim == 3 else cos_p
        sin_k = sin_p.unsqueeze(1) if k.ndim == 3 else sin_p
        q2 = q * cos_q + _rotate_half(q) * sin_q
        k2 = k * cos_k + _rotate_half(k) * sin_k
        return q2.to(q.dtype), k2.to(k.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


__all__ = ["RotaryEmbedding", "_rotate_half"]
