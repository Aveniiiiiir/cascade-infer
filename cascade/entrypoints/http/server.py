"""OpenAI 兼容 HTTP serving（FastAPI + SSE 流式）。

用法（集群）:
    python -m cascade.entrypoints.http.server --model /path/to/DeepSeek-V2-Lite --port 8000

架构：AsyncEngine = 请求队列 + 后台引擎线程（ZOS 三段式 step 循环）+
轮询式流式缓冲（v1：10ms 轮询桥接线程→asyncio；call_soon_threadsafe
桥接后置）。端点：/v1/chat/completions（流/非流）、/v1/completions、
/v1/models。
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from cascade.entrypoints.http.protocol import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    UsageInfo,
)
from cascade.entrypoints.llm import LLM
from cascade.sampling_params import SamplingParams

_POLL_INTERVAL = 0.01  # 流式轮询（v1 桥接）


class AsyncEngine:
    """LLMEngine 的异步门面：后台线程跑引擎循环，请求经队列注入。"""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self._engine = llm._engine
        self._queue: queue.Queue[tuple[str, list[int], SamplingParams]] = queue.Queue()
        self._lock = threading.Lock()
        self._buffers: dict[str, list] = {}  # rid → [RequestOutput]
        self._finished: dict[str, str] = {}  # rid → finish_reason
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._engine_loop, daemon=True, name="cascade-engine")
        self._thread.start()

    # ------------------------------------------------------------------ engine loop

    def _engine_loop(self) -> None:
        while not self._stop.is_set():
            # 1. 收编新请求
            drained = False
            try:
                while True:
                    rid, ids, sp = self._queue.get_nowait()
                    self._engine.add_request(rid, ids, sp)
                    drained = True
            except queue.Empty:
                pass
            # 2. 引擎步进（有活才跑）
            has_work = self._engine.scheduler.has_unfinished_requests or self._engine._so_staging
            if not has_work:
                if not drained:
                    time.sleep(0.005)
                continue
            try:
                outputs = self._engine.step()
            except Exception as e:  # 引擎异常不杀线程（请求级隔离原则外层兜底）
                import logging

                logging.getLogger("cascade.http").exception("engine step failed: %s", e)
                outputs = []
            for ro in outputs:
                with self._lock:
                    self._buffers.setdefault(ro.request_id, []).append(ro)
                    if ro.finish_reason is not None:
                        self._finished[ro.request_id] = ro.finish_reason
        # 收尾冲刷
        for ro in self._engine._post_process_prev():
            with self._lock:
                self._buffers.setdefault(ro.request_id, []).append(ro)
                if ro.finish_reason is not None:
                    self._finished[ro.request_id] = ro.finish_reason

    # ------------------------------------------------------------------ public

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> str:
        rid = f"chat-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._buffers[rid] = []
        self._queue.put((rid, prompt_ids, params))
        return rid

    def abort(self, rid: str) -> None:
        self._engine.abort(rid)

    def iter_outputs(self, rid: str) -> AsyncGenerator:
        """轮询式流式产出（v1 线程→asyncio 桥接）。"""

        async def gen() -> AsyncGenerator:
            while True:
                with self._lock:
                    chunk = self._buffers.get(rid, [])
                    self._buffers[rid] = []
                    finish = self._finished.get(rid)
                for ro in chunk:
                    yield ro
                if finish is not None:
                    return
                await _sleep_async(_POLL_INTERVAL)

        return gen()

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


async def _sleep_async(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


# ------------------------------------------------------------------ app


def _so_args_from_request(req: ChatCompletionRequest) -> dict | None:
    """response_format={"type": "json_schema", "json_schema": {...}} → so_args。"""
    rf = req.response_format
    if not rf:
        return None
    if rf.get("type") == "json_schema":
        schema = rf.get("json_schema") or {}
        if isinstance(schema, dict) and "schema" in schema:
            schema = schema["schema"]
        return {"schema_type": "json", "schema": schema}
    if rf.get("type") == "regex":
        return {"schema_type": "regex", "schema": rf.get("pattern", "")}
    return None


def _chat_prompt_ids(llm: LLM, messages: list[ChatMessage]) -> list[int]:
    tok = llm.tokenizer
    try:
        text = tok.apply_chat_template(
            [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = "".join(f"{m.role}: {m.content}\n" for m in messages)
    return tok.encode(text)


def create_app(llm: LLM) -> FastAPI:
    app = FastAPI(title="Cascade OpenAI-compatible server")
    engine = AsyncEngine(llm)
    model_name = getattr(llm, "_model_dir", "cascade") or "cascade"

    @app.get("/v1/models")
    def list_models() -> dict:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        ids = _chat_prompt_ids(llm, req.messages)
        sp = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens or 256,
            seq_seed=req.seed if req.seed is not None else 0,
            stop_token_ids=(llm.eos_token_id,),
            so_args=_so_args_from_request(req),
        )
        rid = engine.submit(ids, sp)
        created = int(time.time())
        if req.stream:
            return EventSourceResponse(_stream_chat(engine, llm, rid, req.model or model_name, created))
        # 非流式：收齐
        tokens: list[int] = []
        finish = "stop"
        async for ro in engine.iter_outputs(rid):
            tokens.extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                finish = ro.finish_reason
        text = llm.tokenizer.decode(tokens)
        return ChatCompletionResponse(
            id=rid,
            created=created,
            model=req.model or model_name,
            choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=text), finish_reason=finish)],
            usage=UsageInfo(prompt_tokens=len(ids), completion_tokens=len(tokens)),
        )

    async def _stream_chat(engine: AsyncEngine, llm: LLM, rid: str, model: str, created: int):
        import json

        first = True
        acc_tokens: list[int] = []
        acc_text = ""
        finish = "stop"
        async for ro in engine.iter_outputs(rid):
            acc_tokens.extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                finish = ro.finish_reason
            new_text = llm.tokenizer.decode(acc_tokens)
            delta = new_text[len(acc_text) :] if len(new_text) >= len(acc_text) else new_text
            acc_text = new_text
            delta_obj = (
                ChatCompletionChunkDelta(role="assistant") if first else ChatCompletionChunkDelta()
            )
            delta_obj.content = delta
            first = False
            chunk = ChatCompletionChunk(
                id=rid,
                created=created,
                model=model,
                choices=[ChatCompletionChunkChoice(delta=delta_obj, finish_reason=None)],
            )
            yield {"data": chunk.model_dump_json()}
        chunk = ChatCompletionChunk(
            id=rid,
            created=created,
            model=model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(), finish_reason=finish)],
        )
        yield {"data": chunk.model_dump_json()}
        yield {"data": "[DONE]"}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        if len(prompts) != 1:
            return JSONResponse({"error": "仅支持单 prompt"}, status_code=400)
        ids = llm.tokenizer.encode(prompts[0])
        sp = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens or 256,
            seq_seed=req.seed if req.seed is not None else 0,
            stop_token_ids=(llm.eos_token_id,),
        )
        rid = engine.submit(ids, sp)
        tokens: list[int] = []
        finish = "stop"
        async for ro in engine.iter_outputs(rid):
            tokens.extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                finish = ro.finish_reason
        return {
            "id": rid,
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model or model_name,
            "choices": [
                {"index": 0, "text": llm.tokenizer.decode(tokens), "finish_reason": finish}
            ],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade OpenAI-compatible server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    args = parser.parse_args()
    import uvicorn

    llm = LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    app = create_app(llm)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
