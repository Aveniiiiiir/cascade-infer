"""HTTP serving 测试（torch + fastapi；tiny 随机模型 + 假 LLM 门面）。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cascade.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from cascade.engine.llm_engine import LLMEngine
from cascade.entrypoints.http.server import create_app
from cascade.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM


class _StubTokenizer:
    """极简 tokenizer：char ↔ token id 往返（服务器路径验证足够）。"""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) % 400 + 1 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # 仅拼接 content（不加 role 前缀）——否则 prompt_tokens 与断言不符；
        # 真实 tokenizer 的模板由 transformers 提供
        return "".join(m["content"] for m in messages)


class _FakeLLM:
    """server 只用 duck typing（tokenizer/_engine/eos_token_id）。"""

    def __init__(self, engine: LLMEngine) -> None:
        self._engine = engine
        self.eos_token_id = 3
        self.tokenizer = _StubTokenizer()
        self._model_dir = "tiny-test"


def _make_client():
    torch.manual_seed(0)
    cfg = ModelConfig(
        model_type="deepseek_v2", max_model_len=256, vocab_size=512, num_layers=2,
        hidden_size=64, num_attention_heads=4, q_lora_rank=32, kv_lora_rank=16,
        qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16, num_kv_heads=1,
        intermediate_size=32, moe_intermediate_size=32, first_k_dense_replace=1,
        n_routed_experts=8, n_shared_experts=2, num_experts_per_tok=4,
        scoring_func="sigmoid", topk_method="noaux_tc", rms_norm_eps=1e-5,
    )
    engine_cfg = EngineConfig(
        model=cfg, cache=CacheConfig(block_size=8),
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=256),
    )
    engine_cfg.validate()
    model = DeepseekV2ForCausalLM(cfg, device="cpu").to("cpu", torch.float32)
    engine = LLMEngine(engine_cfg, device="cpu", dtype=torch.float32, model=model, model_cfg=cfg)
    app = create_app(_FakeLLM(engine))
    return TestClient(app), app


def test_models_endpoint() -> None:
    client, app = _make_client()
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "tiny-test"
    app.state  # keep refs


def test_chat_completions_non_stream() -> None:
    client, app = _make_client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "tiny-test",
            "messages": [{"role": "user", "content": "abc"}],
            "max_tokens": 6,
            "temperature": 0.0,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) >= 0
    assert body["usage"]["completion_tokens"] == 6
    assert body["usage"]["prompt_tokens"] == 3
    assert body["choices"][0]["finish_reason"] == "length"


def test_chat_completions_stream() -> None:
    client, app = _make_client()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "tiny-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        datas = []
        for line in r.iter_lines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                datas.append(payload)
        assert datas[-1] == "[DONE]"
        chunks = [d for d in datas if d != "[DONE]"]
        assert len(chunks) >= 2  # 至少首 delta + finish chunk
        import json

        first = json.loads(chunks[0])
        assert first["choices"][0]["delta"].get("role") == "assistant"
        last = json.loads(chunks[-1])
        assert last["choices"][0]["finish_reason"] == "length"


def test_metrics_endpoint() -> None:
    """/metrics：Prometheus 文本格式（计数/队列/KV 池）。"""
    client, app = _make_client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "tiny-test",
            "messages": [{"role": "user", "content": "abc"}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
    )
    assert r.status_code == 200
    m = client.get("/metrics")
    assert m.status_code == 200
    assert m.headers["content-type"].startswith("text/plain")
    text = m.text
    for key in (
        "cascade_requests_total",
        "cascade_tokens_generated_total",
        "cascade_engine_steps_total",
        "cascade_kv_blocks",
        "cascade_waiting_seqs",
        "cascade_running_seqs",
    ):
        assert key in text, f"missing metric {key}"
    assert 'finish="length"' in text  # 上方请求以 length 收尾
    app.state  # keep refs


def test_body_too_large_rejected() -> None:
    """评审 #3：请求体 > 10MB → 413（uvicorn 默认不限）。"""
    client, app = _make_client()
    big = b'{"messages":[{"role":"user","content":"' + b"x" * (11 * 2**20) + b'"}]}'
    r = client.post("/v1/chat/completions", content=big,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    app.state  # keep refs


def test_completions_endpoint() -> None:
    client, app = _make_client()
    r = client.post(
        "/v1/completions",
        json={"model": "tiny-test", "prompt": "xy", "max_tokens": 4, "temperature": 0.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "length"
