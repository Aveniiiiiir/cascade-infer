"""SOMatchersBuilder：线程池异步编译（DeepInfer so_matcher_builder.py:19）。

submit() 非阻塞入队（编译在后台线程）；pump() 轮询收割完成的
SOMatcher —— **编译不阻塞调度器主循环**；编译失败/超时 → failed 状态
（请求判死 FINISHED_SO_FAILED，引擎继续）。
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from cascade.common.logger import get_logger
from cascade.structured_output.compiler import Grammar, GrammarCompiler
from cascade.structured_output.matcher import (
    SO_STATUS_FAILED,
    GrammarBackend,
    SOMatcher,
    StubGrammarBackend,
)

_log = get_logger("structured_output.builder")


@dataclass(slots=True)
class SOResult:
    """一次 submit 的收割产物。"""

    request_id: str
    matcher: SOMatcher | None  # None = 失败
    error: str | None = None


class SOMatchersBuilder:
    __slots__ = ("_compiler", "_pool", "_pending", "_lock")

    def __init__(self, max_workers: int = 2) -> None:
        self._compiler = GrammarCompiler()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ submit

    def submit(self, request_id: str, so_args: dict, vocab_size: int) -> bool:
        """非阻塞提交编译。同 rid 重复提交为幂等 no-op。"""
        with self._lock:
            if request_id in self._pending:
                return True
            future = self._pool.submit(self._build_one, request_id, so_args, vocab_size)
            self._pending[request_id] = future
        return True

    def _build_one(self, request_id: str, so_args: dict, vocab_size: int) -> SOResult:
        try:
            grammar: Grammar = self._compiler.compile(so_args)
            backend: GrammarBackend
            if grammar.compiled is not None:
                from cascade.structured_output.matcher import XGrammarBackend

                backend = XGrammarBackend(grammar.compiled, vocab_size)
            else:
                backend = StubGrammarBackend(grammar.schema)
            return SOResult(request_id, SOMatcher(backend, vocab_size))
        except Exception as e:  # 失败隔离：只判死本请求（DeepInfer 语义）
            _log.warning("SO compile failed for %s: %s", request_id, e)
            return SOResult(request_id, None, error=str(e))

    # ------------------------------------------------------------------ pump

    def pump(self) -> list[SOResult]:
        """收割已完成的编译结果（非阻塞；编完的 seq 才入调度）。"""
        results: list[SOResult] = []
        with self._lock:
            done_ids = [rid for rid, f in self._pending.items() if f.done()]
        for rid in done_ids:
            with self._lock:
                future = self._pending.pop(rid)
            results.append(future.result())
        return results

    @property
    def num_pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


__all__ = ["SOMatchersBuilder", "SOResult"]
