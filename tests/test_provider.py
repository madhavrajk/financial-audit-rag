"""Provider-neutral behaviour: model names, request and embedder fallbacks, readiness checks."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import openai
import pytest
from openai import AsyncOpenAI

from app.adapters.openai_compat import (
    QUERY_INSTRUCTION,
    OpenAICompatEmbedder,
    OpenAICompatLLM,
    with_retries,
)
from app.core.agent import Agent
from app.core.config import Settings
from app.core.errors import UpstreamError
from app.core.ports import Usage, model_key
from app.core.pricing import cost_usd
from app.tools import ToolBox


@pytest.mark.parametrize(
    "name",
    [
        "qwen3-embedding-8b",
        "Qwen/Qwen3-Embedding-8B",
        "qwen3-embedding:8b",
        "org/models/qwen3-embedding-8b",
    ],
)
def test_model_key_is_provider_neutral(name):
    assert model_key(name) == "qwen3-embedding-8b"


def test_pricing_matches_across_provider_names():
    usage = Usage(prompt_tokens=1_000_000)
    assert cost_usd("openai/gpt-oss-120b", usage) == cost_usd("gpt-oss-120b", usage) > 0


class FakeEmbeddings:
    def __init__(self, dims_returned: int, reject_dimensions: bool = False) -> None:
        self.dims_returned = dims_returned
        self.reject_dimensions = reject_dimensions
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.reject_dimensions and "dimensions" in kwargs:
            req = httpx.Request("POST", "http://test/embeddings")
            raise openai.BadRequestError(
                "unsupported parameter: dimensions",
                response=httpx.Response(400, request=req),
                body=None,
            )
        data = [
            SimpleNamespace(index=i, embedding=[1.0] * self.dims_returned)
            for i in range(len(kwargs["input"]))
        ]
        return SimpleNamespace(data=data)


def embedder(fake: FakeEmbeddings, model: str = "qwen3-embedding-8b") -> OpenAICompatEmbedder:
    client: Any = SimpleNamespace(embeddings=fake)
    return OpenAICompatEmbedder(client, model=model, dims=4)


async def test_embedder_drops_rejected_dimensions_and_truncates():
    fake = FakeEmbeddings(dims_returned=8, reject_dimensions=True)
    e = embedder(fake)
    [vec] = await e.embed(["page"], is_query=False)
    assert len(vec) == 4 and np.isclose(np.linalg.norm(vec), 1.0)
    await e.embed(["again"], is_query=False)
    assert "dimensions" not in fake.calls[-1]  # remembered; not retried on every call


async def test_embedder_rejects_too_small_vectors():
    with pytest.raises(UpstreamError):
        await embedder(FakeEmbeddings(dims_returned=2)).embed(["page"], is_query=False)


async def test_query_instruction_only_for_qwen3():
    qwen = FakeEmbeddings(dims_returned=4)
    await embedder(qwen).embed(["q"], is_query=True)
    assert qwen.calls[0]["input"][0].startswith(QUERY_INSTRUCTION)
    other = FakeEmbeddings(dims_returned=4)
    await embedder(other, model="text-embedding-3-small").embed(["q"], is_query=True)
    assert other.calls[0]["input"] == ["q"]


def chat_llm(
    sent: list[dict[str, Any]],
    reject: dict[str, tuple[int, str]] | None = None,
    reply: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> OpenAICompatLLM:
    """The real SDK client against a fake server that records each request, rejects requests
    containing a field in `reject`, and otherwise answers with `reply(request)`."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        for field, (status, message) in (reject or {}).items():
            if field in body:
                return httpx.Response(status, json={"error": {"message": message}})
        message = reply(body) if reply else {"role": "assistant", "content": "ok"}
        return httpx.Response(200, json={"choices": [{"index": 0, "message": message}]})

    client = AsyncOpenAI(
        api_key="k",
        base_url="http://test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return OpenAICompatLLM(
        client,
        default_model="gpt-5-mini",
        temperature=0.0,
        max_tokens=100,
        reasoning_effort="low",
        max_concurrency=2,
    )


HI: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
OPENAI_REASONING_MODEL = {  # OpenAI's rejections for GPT-5 and o-series models
    "max_tokens": (
        400,
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead.",
    ),
    "temperature": (
        400,
        "Unsupported value: 'temperature' does not support 0.0 with this model. "
        "Only the default (1) value is supported.",
    ),
}


async def test_llm_adapts_to_openai_reasoning_models():
    sent: list[dict[str, Any]] = []
    llm = chat_llm(sent, OPENAI_REASONING_MODEL)
    assert (await llm.complete(HI)).content == "ok"
    assert sent[-1]["max_completion_tokens"] == 100 and sent[-1]["reasoning_effort"] == "low"
    assert "max_tokens" not in sent[-1] and "temperature" not in sent[-1]
    before = len(sent)
    await llm.complete(HI)
    assert len(sent) == before + 1  # remembered per model: no second round of 400s


@pytest.mark.parametrize(
    "field,message",
    [
        ("reasoning_effort", "Unrecognized request argument supplied: reasoning_effort"),
        ("user", 'Invalid JSON payload received. Unknown name "user": Cannot find field.'),
    ],
)
async def test_llm_drops_optional_fields_a_provider_rejects(field, message):
    sent: list[dict[str, Any]] = []
    await chat_llm(sent, {field: (400, message)}).complete(HI, user="session-1")
    assert field not in sent[-1] and sent[-1]["temperature"] == 0.0


async def test_tool_call_extras_are_sent_back(store, index):
    """Gemini 3 signs its tool calls and rejects a next turn that drops the signature."""
    signature = {"google": {"thought_signature": "sig"}}

    def reply(body: dict[str, Any]) -> dict[str, Any]:
        if any(m["role"] == "tool" for m in body["messages"]):
            return {"role": "assistant", "content": "Done."}
        call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "calculate", "arguments": '{"expressions": {"x": "1 + 1"}}'},
            "extra_content": signature,
        }
        return {"role": "assistant", "content": None, "tool_calls": [call]}

    sent: list[dict[str, Any]] = []
    agent = Agent(chat_llm(sent, reply=reply), ToolBox(store, index), store.schema_description())
    await agent.run("What is 1 + 1?")
    [turn] = [m for m in sent[-1]["messages"] if m.get("tool_calls")]
    assert turn["tool_calls"][0]["extra_content"] == signature


async def test_unknown_model_is_a_clear_error():
    reject = {"model": (404, "The model `gpt-5-mni` does not exist or you do not have access.")}
    with pytest.raises(UpstreamError, match=r"check the model ID.*: The model `gpt-5-mni` does"):
        await chat_llm([], reject).complete(HI)


def rate_limited(message: str) -> openai.RateLimitError:
    """A 429 shaped like Gemini's: the wait is only in the message, with no Retry-After."""
    req = httpx.Request("POST", "http://test/v1/chat/completions")
    body = [{"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED"}}]
    return openai.RateLimitError("429", response=httpx.Response(429, request=req), body=body)


@pytest.mark.parametrize(
    "message,wait",
    [
        ("Requests per minute exceeded on Free Tier. Please retry in 20.25s.", 20.25),
        ("Rate limit reached. Please try again in 450ms.", 0.45),
    ],
)
async def test_rate_limit_waits_as_long_as_the_provider_asks(monkeypatch, message, wait):
    waits: list[float] = []

    async def no_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    errors = [rate_limited(message)]

    async def call() -> str:
        if errors:
            raise errors.pop()
        return "ok"

    assert await with_retries(call) == "ok"
    assert waits == [wait]


@pytest.mark.parametrize(
    "message,wait",
    [
        # Gemini's real daily-quota message, with a Go-style duration.
        ("Requests per day limit exceeded on Free Tier. Please retry in 4h30m34.57s.", "4.5 hours"),
        ("Request limit reached. Please try again in 1m25.632s.", "1 min"),
    ],
)
async def test_quota_fails_fast_with_the_providers_words(message, wait):
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise rate_limited(message)

    with pytest.raises(UpstreamError, match=rf"used up \(it asks to wait {wait}\): Request"):
        await with_retries(call)
    assert calls == 1


def settings(**kw: Any) -> Settings:
    return Settings(_env_file=None, **kw)  # ignore a developer's local .env


def test_provider_problems():
    unconfigured = settings(llm_model="").provider_problems()
    assert len(unconfigured) == 1 and "aistudio.google.com/apikey" in unconfigured[0]
    gemini = settings(llm_model="m", llm_base_url="https://generativelanguage.googleapis.com/v1/")
    assert "aistudio.google.com/apikey" in gemini.provider_problems()[0]
    unknown = settings(llm_model="m", llm_base_url="https://llm.example.com/v1")
    assert "llm.example.com" in unknown.provider_problems()[0]
    assert "LLM_API_KEY" in settings(llm_model="m").provider_problems()[0]
    assert (
        settings(llm_model="m", llm_base_url="http://localhost:11434/v1").provider_problems() == []
    )
    assert settings(llm_model="m", llm_api_key="k").provider_problems() == []
