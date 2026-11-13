"""语法编译器：so_args → 后端无关的 Grammar 对象（DeepInfer compiler.py:54-63）。

so_args 契约（与 DeepInfer SamplingParams.so_args 一致）:
    {"schema_type": "json",  "schema": <json-schema dict 或 str>}
    {"schema_type": "regex", "schema": <pattern str>}
    {"schema_type": "ebnf",  "schema": <grammar str>}

无 xgrammar 的环境：regex/ebnf 走 StubGrammar（确定性逐步约束，
引擎路径测试用）；json 需要 xgrammar（本地明确报错而不是静默错行为）。

审查 #10：缓存此前无上限且被 builder 的 worker 线程并发读写——
互不相同的 schema 可无界增长（内存 DoS 面，DeepInfer 用
DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES 同位防御）。现为 OrderedDict LRU
（容量上限）+ 锁保护的 double-check；编译本身在锁外（可秒级）。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

try:
    import xgrammar as _xgrammar

    HAS_XGRAMMAR = True
except ImportError:  # pragma: no cover - Windows 开发机
    _xgrammar = None  # type: ignore[assignment]
    HAS_XGRAMMAR = False

from cascade.common.logger import get_logger

_log = get_logger("structured_output.compiler")

#: 缓存条目上限（LRU 驱逐最旧）
_CACHE_LIMIT = 256


@dataclass(slots=True)
class Grammar:
    """编译产物（后端无关）：kind + schema + 编译好的后端对象（若有）。"""

    schema_type: str  # json | regex | ebnf
    schema: str  # 规范化后的 schema 文本
    compiled: object | None = None  # xgrammar.CompiledGrammar（真后端）


class GrammarCompiler:
    """dispatch_and_compile（json strict / regex / ebnf 三路分发）。"""

    __slots__ = ("_cache", "_lock")

    def __init__(self) -> None:
        self._cache: OrderedDict[str, Grammar] = OrderedDict()
        self._lock = Lock()

    def compile(self, so_args: dict) -> Grammar:
        schema_type = so_args.get("schema_type")
        schema = so_args.get("schema")
        if schema_type not in ("json", "regex", "ebnf"):
            raise ValueError(f"不支持的 schema_type {schema_type!r}")
        if schema is None:
            raise ValueError("so_args 缺少 schema")
        if isinstance(schema, dict):  # json schema 允许 dict 形式
            import json

            schema = json.dumps(schema)
        key = f"{schema_type}:{schema}"
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)  # LRU touch
                return cached
        g = self._compile(schema_type, schema)  # 慢路径不加锁
        with self._lock:
            if key in self._cache:  # 并发竞态：用先完成者
                self._cache.move_to_end(key)
                return self._cache[key]
            self._cache[key] = g
            while len(self._cache) > _CACHE_LIMIT:
                self._cache.popitem(last=False)
        return g

    def _compile(self, schema_type: str, schema: str) -> Grammar:
        if HAS_XGRAMMAR:
            # xgrammar ≥0.2：GrammarCompiler 需要 tokenizer_info（无默认构造）。
            # 无真实 tokenizer 时用空 vocab 的 TokenizerInfo 占位——仅用于
            # 编译期（mask 宽度由运行时 vocab_size 决定，与此无关）。
            info = _xgrammar.TokenizerInfo([])
            compiler = _xgrammar.GrammarCompiler(info)
            if schema_type == "json":
                compiled = compiler.compile_json_schema(schema, strict_mode=True)
            elif schema_type == "regex":
                compiled = compiler.compile_regex(schema)
            else:
                compiled = compiler.compile_grammar(schema)
            return Grammar(schema_type=schema_type, schema=schema, compiled=compiled)
        # 无 xgrammar：regex/ebnf 走确定性 stub（引擎路径测试）；
        # json 明确报错（行为差异大，不静默降级）
        if schema_type == "json":
            raise RuntimeError(
                "JSON schema 约束需要 xgrammar（pip install xgrammar）"
                "——本环境不可用，拒绝静默降级"
            )
        _log.info("xgrammar 不可用：schema_type=%s 使用 StubGrammar", schema_type)
        return Grammar(schema_type=schema_type, schema=schema, compiled=None)


def has_xgrammar() -> bool:
    return HAS_XGRAMMAR


__all__ = ["Grammar", "GrammarCompiler", "has_xgrammar"]
