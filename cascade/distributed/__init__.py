"""分布式层：EP 专家并行分发（NCCL）+ EPLB + KV 传输接口（EPD 后置）。

NCCL 相关全部惰性导入 + is_available 守卫（Windows 导入不报错，
DeepInfer 的 posix_ipc 依赖不引入）。
"""

from cascade.common.logger import get_logger

_log = get_logger("distributed")

try:
    import torch.distributed as dist

    _HAS_DIST = dist.is_available()
except Exception:  # pragma: no cover
    dist = None  # type: ignore[assignment]
    _HAS_DIST = False


def dist_available() -> bool:
    return _HAS_DIST


__all__ = ["dist_available", "dist"]
