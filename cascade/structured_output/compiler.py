"""语法编译器：so_args → 后端无关的 Grammar 对象（DeepInfer compiler.py:54-63）。

so_args 契约（与 DeepInfer SamplingParams.so_args 一致）:
    {"schema_type": "json",  "schema": <json-schema dict 或 str>}
    {"schema_type": "regex", "schema": <pattern str>}
    {"schema_type": "ebnf",  "schema": <grammar str>}

无 xgrammar 的环境：regex/ebnf 走 StubGrammar（确定性逐步约束，
引擎路径测试用）；json 需要 xgrammar（本地明确报错而不是静默错行为）。
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import xgrammar as _xgrammar

    HAS_XGRAMMAR = True
except ImportError:  # pragma: no cover - Windows 开发机
    _xgrammar = None  # type: ignore[assignment]
    HAS_XGRAMMAR = False

from cascade.common.logger import get_logger

_log = get_logger("structured_output.compiler")


@dataclass(slots=True)
class Grammar:
    """编译产物（后端无关）：kind + schema + 编译好的后端对象（若有）。"""

    schema_type: str  # json | regex | ebnf
    schema: str  # 规范化后的 schema 文本
    compiled: object | None = None  # xgrammar.CompiledGrammar（真后端）


class GrammarCompiler:
    """dispatch_and_compile（json strict / regex / ebnf 三路分发）。"""

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[str, Grammar] = {}

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
        if key in self._cache:
            return self._cache[key]
        g = self._compile(schema_type, schema)
        self._cache[key] = g
        return g

    def _compile(self, schema_type: str, schema: str) -> Grammar:
        if HAS_XGRAMMAR:
            compiler = _xgrammar.GrammarCompiler()
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
