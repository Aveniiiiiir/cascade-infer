"""基础 Linear（无 bias，DeepSeek 全家 bias=False）。FP8 量化变体 Phase 6。"""

from __future__ import annotations

import torch.nn as nn


class Linear(nn.Linear):
    """bias=False 的标准 Linear；显式暴露 weight 以便加载器直接映射。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__(in_features, out_features, bias=bias)


__all__ = ["Linear"]
