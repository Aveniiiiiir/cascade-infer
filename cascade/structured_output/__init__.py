"""结构化输出（SO）集成层。

移植 DeepInfer structure_output/ 的工程价值（文法引擎本身用 xgrammar）：
- 异步编译不阻塞调度（builder 的 submit/pump）
- per-request 失败隔离（so_status=-1 → FINISHED_SO_FAILED，不崩引擎）
- MTP 多步掩码：试探性接受 draft 填独立掩码后回滚（不动真实状态）
- 掩码 CPU pinned → 异步 H2D 与 logits 计算重叠（runner 侧）

后端接口化：XGrammarBackend（集群真后端）+ StubBackend（本地引擎
路径测试——确定性逐步约束状态机）。状态机：0=未激活 / 1=首步激活 /
2=活跃 / -1=失败（DeepInfer so_status 语义）。
"""

from cascade.structured_output.compiler import GrammarCompiler, has_xgrammar
from cascade.structured_output.matcher import (
    SO_STATUS_ACTIVE,
    SO_STATUS_FAILED,
    SO_STATUS_INACTIVE,
    StubGrammarBackend,
    XGrammarBackend,
)
from cascade.structured_output.builder import SOMatchersBuilder

__all__ = [
    "GrammarCompiler",
    "has_xgrammar",
    "SOMatchersBuilder",
    "StubGrammarBackend",
    "XGrammarBackend",
    "SO_STATUS_INACTIVE",
    "SO_STATUS_ACTIVE",
    "SO_STATUS_FAILED",
]
