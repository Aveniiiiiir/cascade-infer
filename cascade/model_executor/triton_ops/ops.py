"""后端检测与分发设施。

所有算子 wrapper 统一通过 `use_triton(device)` 决定走 Triton kernel 还是
PyTorch 参考路径 —— 引擎在无 triton 的环境（Windows 开发机）自动降级，
ref 路径永远在场（相对 DeepInfer tilelang 绑定的可移植性优势）。
"""

from __future__ import annotations

try:  # triton 官方只发 Linux 轮子
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - Windows/无 GPU 环境
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


def use_triton(device) -> bool:
    """Triton kernel 可用且张量在 CUDA 上。"""
    return HAS_TRITON and str(device).startswith("cuda")


__all__ = ["HAS_TRITON", "use_triton", "triton", "tl"]
