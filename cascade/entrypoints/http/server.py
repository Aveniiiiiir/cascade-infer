"""OpenAI 兼容 HTTP serving（FastAPI + SSE 流式）。

用法（集群）:
    python -m cascade.entrypoints.http.server --model /path/to/DeepSeek-V2-Lite --port 8000

架构：AsyncEngine = 请求队列 + 后台引擎线程（同步序 step 循环）+
轮询式流式缓冲（v1：10ms 轮询桥接线程→asyncio；call_soon_threadsafe
桥接后置）。端点：/v1/chat/completions（流/非流）、/v1/completions、
/v1/models。

审查 #4/#6/#14/#18 修复要点：
- 引擎线程对 add_request / step 的异常**永不死亡**：单请求畸形只判死该
  请求（error 回执），引擎步异常则 fail_all_inflight 后继续；
- _buffers/_finished 在消费方结束（含断连/超时）时清理，杜绝无界增长；
- prompt 空串 / 超长在 handler 层 400 拒绝（不进引擎线程）；
- stop 字符串 tokenize 为 token 序列交给调度器后缀截断；
- 可选 --api-key（Bearer）；默认仍监听 0.0.0.0（集群内网部署假设，
  对外暴露请务必配 api-key 或网络隔离）。
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from cascade.common.logger import get_logger
from cascade.core.sequence import RequestOutput
from cascade.entrypoints.http.protocol import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    UsageInfo,
)
from cascade.entrypoints.llm import LLM
from cascade.sampling_params import SamplingParams

_log = get_logger("http.serving")

_POLL_INTERVAL = 0.01  # 流式轮询（v1 桥接）
_REQUEST_TIMEOUT_S = 600.0  # 单请求 wall-clock 上限（防引擎挂死拖死连接）


def _bad_request(msg: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": msg, "type": "invalid_request_error"}},
        status_code=400,
    )


class AsyncEngine:
    """LLMEngine 的异步门面：后台线程跑引擎循环，请求经队列注入。

    2026-11-18 审计 #2/#4 修复：
    - **abort 串行化**：此前从 asyncio 线程直接调 scheduler.abort，与引擎
      线程的 step 并发——Sequence/block_table 竞态可致崩溃（断言炸 →
      fail_all_inflight 全批陪葬）或块双重释放 → 跨请求静默 KV 串写。
      现改为 abort 队列，由引擎线程在每轮 step 前 drain 执行。
    - **活跃 rid 集合**：消费方退出（断连/超时清理）后 _put 直接丢弃
      迟到输出并止损——旧 setdefault 会复活死键，条目随每步 append
      无界增长（每条 RequestOutput 携带全量输出拷贝）。
    """

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self._engine = llm._engine
        self._queue: queue.Queue[tuple[str, list[int], SamplingParams]] = queue.Queue()
        self._abort_queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._active: set[str] = set()  # 消费方仍在消费的 rid
        self._buffers: dict[str, list] = {}  # rid → [RequestOutput]
        self._finished: dict[str, str] = {}  # rid → finish_reason
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._engine_loop, daemon=True, name="cascade-engine")
        self._thread.start()

    # ------------------------------------------------------------------ engine loop

    def _engine_loop(self) -> None:
        from cascade.engine.llm_engine import EngineStepError

        while not self._stop.is_set():
            # 0. abort 串行化（审计 #2：与 add_request 同一模式——引擎线程
            #    是调度器状态的唯一变更者）
            while True:
                try:
                    rid = self._abort_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._engine.abort(rid)
                except Exception:
                    _log.exception("abort %s failed", rid)
            # 1. 收编新请求（审查 #4：add_request 异常只判死该请求，不杀线程）
            drained = False
            try:
                while True:
                    rid, ids, sp = self._queue.get_nowait()
                    drained = True
                    try:
                        # LLMEngine.add_request 签名：(prompt_token_ids,
                        # sampling_params, request_id)——位置序勿与 (rid, ids, sp) 混淆
                        self._engine.add_request(ids, sp, rid)
                    except Exception as e:
                        _log.warning("add_request rejected %s: %s", rid, e)
                        with self._lock:
                            self._finished[rid] = "error"
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
            except EngineStepError as ese:
                # 审计 #3：引擎层已复位 pending 并携带判死回执——必须 _put
                # 它们，否则在途客户端各自挂满 600s 超时
                _log.exception("engine step failed; failing all in-flight requests")
                outputs = list(ese.receipts)
            except Exception:
                _log.exception("engine step failed; failing all in-flight requests")
                outputs = self._engine.scheduler.fail_all_inflight()
            for ro in outputs:
                self._put(ro)
            # 审计 #10：SO 编译 pending（最长 30s）且本步无产出 → 免 CPU 空转
            if not outputs and self._engine._so_builder.num_pending > 0:
                time.sleep(0.001)
        # 收尾冲刷（判死回执；同步序下无在途采样滞后）
        for ro in self._engine.scheduler.flush_final_outputs():
            self._put(ro)

    def _put(self, ro: RequestOutput) -> None:
        with self._lock:
            alive = ro.request_id in self._active
            if alive:
                self._buffers.setdefault(ro.request_id, []).append(ro)
                if ro.finish_reason is not None:
                    self._finished[ro.request_id] = ro.finish_reason
        if not alive:
            # 审计 #4：消费方已退出（断连/超时清理后）的迟到输出——丢弃并
            # 止损。_put 在引擎线程执行，直接 abort 无跨线程竞态。
            try:
                self._engine.abort(ro.request_id)
            except Exception:
                _log.exception("late-output abort %s failed", ro.request_id)

    # ------------------------------------------------------------------ public

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> str:
        rid = f"chat-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._active.add(rid)
            self._buffers[rid] = []
        self._queue.put((rid, prompt_ids, params))
        return rid

    def abort(self, rid: str) -> bool:
        """投递到引擎线程串行执行（审计 #2——跨线程直调会与 step 竞态）。

        返回值 = 是否为已知活跃请求（异步执行，不等待结果）。
        """
        with self._lock:
            known = rid in self._active
        if known:
            self._abort_queue.put(rid)
        return known

    def iter_outputs(self, rid: str, timeout_s: float = _REQUEST_TIMEOUT_S) -> AsyncGenerator:
        """轮询式流式产出（v1 线程→asyncio 桥接）。

        审查 #6：生成器退出（finish/断连/超时）时清理 _buffers/_finished/
        _active 的 rid 键——否则键随请求数无界累积；超时触发 abort 让引擎
        停止为该请求继续算 KV。

        2026-11-18 审计 #4 补充：非正常退出（断连，未见到 finish）时投递
        abort 止损——断连不取消引擎侧计算，否则引擎会为死连接白算到
        max_tokens 且迟到输出持续复活死键。
        """

        async def gen() -> AsyncGenerator:
            deadline = time.monotonic() + timeout_s
            try:
                while True:
                    with self._lock:
                        chunk = self._buffers.get(rid, [])
                        self._buffers[rid] = []
                        finish = self._finished.get(rid)
                    for ro in chunk:
                        yield ro
                    if finish is not None:
                        return
                    if time.monotonic() > deadline:
                        _log.warning("request %s timed out after %.0fs; aborting", rid, timeout_s)
                        self.abort(rid)
                        return
                    await _sleep_async(_POLL_INTERVAL)
            finally:
                with self._lock:
                    self._buffers.pop(rid, None)
                    finished = self._finished.pop(rid, None) is not None
                    self._active.discard(rid)
                if not finished:
                    # 消费方退出且请求未完成（断连/消费异常）→ 止损
                    self._abort_queue.put(rid)

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


def _stop_token_seqs(llm: LLM, stop: list[str] | None) -> list[tuple[int, ...]] | None:
    """stop 字符串 → token 序列（调度器后缀截断）。空串/编码失败 → None
    表示应拒绝请求（返回哨兵由调用方 400）。"""
    if not stop:
        return []
    seqs: list[tuple[int, ...]] = []
    for s in stop:
        if not isinstance(s, str) or not s:
            return None
        enc = llm.tokenizer.encode(s)
        if not enc:
            return None
        seqs.append(tuple(enc))
    return seqs


def _validate_prompt(llm: LLM, ids: list[int]) -> JSONResponse | None:
    """审查 #4：空/超长 prompt 在 handler 层拒绝，不进引擎线程。

    2026-11-18 审计 #1 配套：`>=`——长度恰等于 max_model_len 的 prompt
    没有为任何生成 token 留位（引擎入队守卫同样会拒），此处先行 400。
    """
    if not ids:
        return _bad_request("prompt tokenize 后为空（messages 内容为空）")
    max_len = llm._engine.cfg.model.max_model_len
    if len(ids) >= max_len:
        return _bad_request(f"prompt {len(ids)} tokens 达到 max_model_len={max_len}（无生成空间）")
    return None


def create_app(llm: LLM, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="Cascade OpenAI-compatible server")
    engine = AsyncEngine(llm)
    model_name = getattr(llm, "_model_dir", "cascade") or "cascade"

    if api_key:
        @app.middleware("http")
        async def _check_api_key(request: Request, call_next: object) -> Response:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[len("Bearer "):] != api_key:
                return JSONResponse(
                    {"error": {"message": "invalid api key", "type": "auth_error"}},
                    status_code=401,
                )
            return await call_next(request)

    @app.get("/v1/models")
    def list_models() -> dict:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> object:
        ids = _chat_prompt_ids(llm, req.messages)
        if (err := _validate_prompt(llm, ids)) is not None:
            return err
        stop_seqs = _stop_token_seqs(llm, req.stop)
        if stop_seqs is None:
            return _bad_request("stop 序列含空串或无法 tokenize")
        try:
            sp = SamplingParams(
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens or 256,
                # 2026-11-18 审计 #9：无 seed 的请求走 SamplingParams 的
                # 每请求随机默认——服务层强制 0 会使同 prompt 跨请求输出
                # 逐位相同（可复现性侧信道 + 多样性归零）
                seq_seed=req.seed,
                stop_token_ids=(llm.eos_token_id,),
                stop_token_seqs=stop_seqs,
                so_args=_so_args_from_request(req),
            )
        except ValueError as e:
            return _bad_request(str(e))
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
        if finish == "error":
            return JSONResponse(
                {"error": {"message": "request rejected by engine", "type": "engine_error"}},
                status_code=500,
            )
        text = llm.tokenizer.decode(tokens)
        return ChatCompletionResponse(
            id=rid,
            created=created,
            model=req.model or model_name,
            choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=text), finish_reason=finish)],
            usage=UsageInfo(
                prompt_tokens=len(ids),
                completion_tokens=len(tokens),
                total_tokens=len(ids) + len(tokens),
            ),
        )

    async def _stream_chat(
        engine: AsyncEngine, llm: LLM, rid: str, model: str, created: int
    ) -> AsyncGenerator[dict, None]:
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
    async def completions(req: CompletionRequest) -> object:
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        if len(prompts) != 1:
            return _bad_request("仅支持单 prompt")
        ids = llm.tokenizer.encode(prompts[0])
        if (err := _validate_prompt(llm, ids)) is not None:
            return err
        try:
            sp = SamplingParams(
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens or 256,
                # 2026-11-18 审计 #9：无 seed 的请求走 SamplingParams 的
                # 每请求随机默认——服务层强制 0 会使同 prompt 跨请求输出
                # 逐位相同（可复现性侧信道 + 多样性归零）
                seq_seed=req.seed,
                stop_token_ids=(llm.eos_token_id,),
            )
        except ValueError as e:
            return _bad_request(str(e))
        rid = engine.submit(ids, sp)
        tokens: list[int] = []
        finish = "stop"
        async for ro in engine.iter_outputs(rid):
            tokens.extend(ro.new_token_ids)
            if ro.finish_reason is not None:
                finish = ro.finish_reason
        if finish == "error":
            return JSONResponse(
                {"error": {"message": "request rejected by engine", "type": "engine_error"}},
                status_code=500,
            )
        return {
            "id": rid,
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model or model_name,
            "choices": [
                {"index": 0, "text": llm.tokenizer.decode(tokens), "finish_reason": finish}
            ],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(tokens),
                "total_tokens": len(ids) + len(tokens),
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade OpenAI-compatible server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument(
        "--api-key",
        default=None,
        help="可选 Bearer token；对外暴露时务必设置（默认 0.0.0.0 假设内网部署）",
    )
    args = parser.parse_args()
    import uvicorn

    llm = LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    app = create_app(llm, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
