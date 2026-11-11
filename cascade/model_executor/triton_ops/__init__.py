"""Triton 算子层：每文件 = wrapper（断言 + 后端分发）+ Triton kernel + `_ref` 纯 PyTorch 参考实现。

对应 DeepInfer 的 `tilelang_ops` 三层模式（wrapper / kernel / pytest）。
Triton 仅在 Linux CUDA 上可用；Windows 开发机自动走 ref 路径，
集群测试对拍两者。
"""

from cascade.model_executor.triton_ops.ops import HAS_TRITON, use_triton

__all__ = ["HAS_TRITON", "use_triton"]
