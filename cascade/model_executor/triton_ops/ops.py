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
    """Triton kernel 可用且张量在 CUDA 上。

    逃生门（CASCADE_DISABLE_TRITON=1）：强制全部走 PyTorch 参考路径——
    新架构（如 Blackwell sm_120）上 Triton 编译异常时，引擎仍可正确运行
    （慢），便于隔离「kernel 问题」与「引擎问题」。图捕获请同时关
    CASCADE_CUDAGRAPH（ref 路径含 .item()，不可捕获）。
    """
    if not str(device).startswith("cuda") or not HAS_TRITON:
        return False
    from cascade.common.env import envs

    return not envs.DISABLE_TRITON


__all__ = ["HAS_TRITON", "use_triton", "triton", "tl"]
