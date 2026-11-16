"""RMSNorm（DeepSeek V2 语义：fp32 计算再回投）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """与 HF DeepseekV2RMSNorm 数值路径一致：
        fp32 归一化 → 转回模型 dtype → 乘 weight（weight 已在模型 dtype）。"""
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


__all__ = ["RMSNorm"]
