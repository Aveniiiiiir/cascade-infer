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

    def apply(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q/k: [..., head_dim]（最后一维 rope）。positions: 同 broadcast 到 [...,] 的绝对位置。

        NeoX rotate_half: x' = x*cos + rotate_half(x)*sin，
        rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])。
        """
        max_pos = int(positions.max().item()) + 1 if positions.numel() else 1
        cos, sin = self.cos_sin(max_pos)
        cos_p = cos[positions]  # [..., dim]
        sin_p = sin[positions]
        q2 = q * cos_p + _rotate_half(q) * sin_p
        k2 = k * cos_p + _rotate_half(k) * sin_p
        return q2.to(q.dtype), k2.to(k.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


__all__ = ["RotaryEmbedding", "_rotate_half"]
