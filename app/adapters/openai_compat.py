"""Adapters for the LLMClient and Embedder ports over any OpenAI-compatible API.

Works with OpenAI, OpenRouter, Together, Groq, DeepInfra, vLLM, Ollama, LM Studio, or anything
else that serves `/chat/completions` (with tool calling) and `/embeddings`.

Retry policy:
  * retry ONLY transient failures: 429, 5xx, timeouts, connection errors;
  * wait as long as the server asks (Retry-After, or "retry in 20s" in the error text), up to a
    minute; a longer wait (a daily quota) fails at once. Otherwise exponential backoff with
    full jitter;
  * never retry 400/401/403/404, since repeating them only burns latency and quota; the
    exception is a 400 rejecting an optional field (see _OPTIONAL_FIELDS), resent without it;
  * a per-process semaphore caps in-flight calls (backpressure instead of a 429 storm).
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import numpy as np
import openai
from openai import AsyncOpenAI

from app.core.errors import UpstreamAuthError, UpstreamError
from app.core.ports import LLMResponse, Message, ToolCall, ToolChoice, Usage, model_key
from app.core.pricing import cost_usd

log = logging.getLogger(__name__)
T = TypeVar("T")

_THINK = re.compile(r"<think>.*?</think>", re.S)
QUERY_INSTRUCTION = (
    "Instruct: Given a question about a company's annual report (Form 10-K), retrieve the "
    "filing pages that contain the answer\nQuery: "
)


# A wait suggested in the error text, as a Go-style duration: "Please retry in 20.3s." or
# "Please retry in 4h30m34.5s." (Gemini sends no Retry-After header), "Please try again in
# 450ms." (OpenAI).
_RETRY_IN = re.compile(r"(?:retry|try again) in ((?:\d+(?:\.\d+)?\s*(?:h|ms|m|s))+)\b", re.I)
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(h|ms|m|s)", re.I)
_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
MAX_WAIT_S = 60.0  # a longer suggested wait (a daily quota) fails the request instead


def _provider_message(exc: openai.APIStatusError) -> str:
    """The provider's own error text, without the JSON wrapper (Gemini wraps it in a list)."""
    body: Any = exc.body
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        body = body.get("error", body)
    message = body.get("message") if isinstance(body, dict) else None
    return str(message).strip() if message else str(exc)


def _retry_after_s(exc: openai.APIStatusError) -> float | None:
    """The provider's suggested wait: the Retry-After header, else the one in its message."""
    value = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        if value is not None:
            return float(value)
    except ValueError:
        pass
    m = _RETRY_IN.search(_provider_message(exc))
    if m is None:
        return None
    return sum(float(n) * _SECONDS[unit.lower()] for n, unit in _DURATION_PART.findall(m[1]))


async def with_retries(
    fn: Callable[[], Awaitable[T]], *, attempts: int = 4, base_delay_s: float = 1.0
) -> T:
    last: Exception | None = None
    tries = 0
    for attempt in range(attempts):
        tries += 1
        try:
            return await fn()
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise UpstreamAuthError(
                "The model provider rejected the API key (check LLM_API_KEY / EMBED_API_KEY)."
            ) from exc
        except openai.RateLimitError as exc:
            last, delay = exc, _retry_after_s(exc)
        except openai.APIStatusError as exc:
            if exc.status_code < 500:
                raise  # 400/404/422: a bug in the request; retrying cannot help
            last, delay = exc, _retry_after_s(exc)
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            last, delay = exc, None
        if attempt == attempts - 1 or (delay is not None and delay > MAX_WAIT_S):
            break
        sleep_s = delay if delay is not None else random.uniform(0, base_delay_s * 2**attempt)
        log.warning(
            "transient upstream error (%s); retry %d in %.1fs",
            type(last).__name__,
            attempt + 1,
            sleep_s,
        )
        await asyncio.sleep(sleep_s)
    retry_after = None
    detail = str(last)
    if isinstance(last, openai.APIStatusError):
        ra = _retry_after_s(last)
        retry_after = int(ra) if ra is not None else None
        detail = _provider_message(last)
    if isinstance(last, openai.RateLimitError) and retry_after and retry_after > MAX_WAIT_S:
        wait = (
            f"{retry_after / 3600:.1f} hours" if retry_after >= 3600 else f"{retry_after // 60} min"
        )
        reason = f"The model provider's quota for this API key is used up (it asks to wait {wait})"
    elif isinstance(last, openai.RateLimitError):
        reason = (
            "The model provider is rate-limiting this API key (free tiers allow only a few "
            "requests per minute)"
        )
    else:
        reason = f"The model provider is unavailable after {tries} attempts"
    raise UpstreamError(f"{reason}: {detail}", retry_after_s=retry_after or 10)


class AdaptiveLimiter:
    """Concurrency limit that adapts to the provider's rate limit (AIMD, as in TCP).

    A fixed limit is wrong in one direction or the other: in the load test, 8 in-flight
    calls queued needlessly (125 answers/min plateau) while 32 overran the provider's rate
    limit at 50 users (54% of requests failed). AIMD halves the limit on every 429 and adds
    one back after `limit` consecutive successes, so the process settles just under the real
    ceiling, and excess requests wait instead of failing.
    """

    def __init__(self, max_limit: int, min_limit: int = 2) -> None:
        self.max_limit = max_limit
        self.min_limit = min(min_limit, max_limit)
        self.limit = max_limit
        self.in_flight = 0
        self._successes = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: self.in_flight < self.limit)
            self.in_flight += 1

    async def __aexit__(self, *exc: object) -> None:
        async with self._cond:
            self.in_flight -= 1
            self._cond.notify_all()

    def on_rate_limited(self) -> None:
        new = max(self.min_limit, self.limit // 2)
        if new != self.limit:
            log.warning("rate limited: concurrency %d -> %d", self.limit, new)
        self.limit, self._successes = new, 0

    def on_success(self) -> None:
        self._successes += 1
        if self._successes >= self.limit and self.limit < self.max_limit:
            self.limit += 1
            self._successes = 0


# Optional request fields some models reject with a 400, each with a word the rejection
# contains. A rejected field is remembered per model and the call resent without it:
# non-reasoning models reject `reasoning_effort`; OpenAI reasoning models (GPT-5, o-series)
# accept only the default temperature and want `max_completion_tokens` instead of `max_tokens`;
# some providers don't know `user`. Most specific words first; `user` is only a routing hint.
_OPTIONAL_FIELDS = {
    "max_tokens": "max_completion_tokens",
    "temperature": "temperature",
    "reasoning_effort": "reasoning",
    "user": "user",
}


def _rejected_field(exc: openai.BadRequestError, request: dict[str, Any]) -> str | None:
    message = str(exc).lower()
    return next(
        (f for f, word in _OPTIONAL_FIELDS.items() if f in request and word in message), None
    )


def _without(kwargs: dict[str, Any], rejected: set[str]) -> dict[str, Any]:
    request = {k: v for k, v in kwargs.items() if k not in rejected}
    if "max_tokens" in rejected:
        request["max_completion_tokens"] = kwargs["max_tokens"]
    return request


class OpenAICompatLLM:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        default_model: str,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None,
        max_concurrency: int,
    ) -> None:
        self._client = client
        self._default_model = default_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._rejected: dict[str, set[str]] = {}  # model -> optional fields it rejected
        self.limiter = AdaptiveLimiter(max_concurrency)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
        model: str | None = None,
        max_tokens: int | None = None,
        user: str | None = None,
    ) -> LLMResponse:
        model = model or self._default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if user:
            # Standard OpenAI field; providers with prompt-prefix caching use it to route a
            # session's requests to the same replica, so the cache is hit more often.
            kwargs["user"] = user
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        rejected = self._rejected.setdefault(model, set())

        async def call() -> Any:
            async with self.limiter:
                while True:  # each pass drops one more rejected field, so this terminates
                    request = _without(kwargs, rejected)
                    try:
                        resp = await self._client.chat.completions.create(**request)
                        self.limiter.on_success()
                        return resp
                    except openai.RateLimitError:
                        self.limiter.on_rate_limited()
                        raise
                    except openai.BadRequestError as exc:
                        field = _rejected_field(exc, request)
                        if field is None:
                            raise
                        log.info("%s rejected %s; resending without it", model, field)
                        rejected.add(field)

        try:
            resp = await with_retries(call)
        except openai.APIStatusError as exc:  # any other 4xx: a wrong model ID, URL or request
            hint = " (check the model ID and LLM_BASE_URL)" if exc.status_code == 404 else ""
            raise UpstreamError(
                f"The model provider rejected the request{hint}: {_provider_message(exc)}"
            ) from exc
        choice = resp.choices[0]
        msg = choice.message
        usage = Usage()
        if resp.usage is not None:
            details = getattr(resp.usage, "prompt_tokens_details", None)
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
                cached_tokens=(getattr(details, "cached_tokens", None) or 0) if details else 0,
            )
        content = _THINK.sub("", msg.content).strip() if msg.content else None
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "{}",
                extra=tc.model_extra or {},
            )
            for tc in (msg.tool_calls or [])
            if getattr(tc, "function", None) is not None
        ]
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            cost_usd=cost_usd(model, usage),
        )


class OpenAICompatEmbedder:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        dims: int,
        batch_size: int = 32,
        max_concurrency: int = 4,
    ) -> None:
        self._client = client
        self.model_name = model
        self._dims = dims
        self._batch_size = batch_size
        self._sem = asyncio.Semaphore(max_concurrency)
        # Qwen3-Embedding is instruction-aware on the query side only; other models are not.
        self._query_instruction = (
            QUERY_INSTRUCTION if model_key(model).startswith("qwen3-embedding") else ""
        )
        self._send_dimensions = True  # cleared if the provider rejects the `dimensions` field

    def _fit(self, vec: list[float]) -> list[float]:
        """Matryoshka truncation for providers that ignore `dimensions`: keep the first
        `dims` components and re-normalise (what `dimensions` does server-side)."""
        if len(vec) == self._dims:
            return vec
        if len(vec) < self._dims:
            raise UpstreamError(
                f"{self.model_name} returned {len(vec)}-dim vectors; EMBED_DIMS={self._dims}"
            )
        v = np.asarray(vec[: self._dims], dtype=np.float32)
        return (v / (np.linalg.norm(v) or 1.0)).tolist()

    async def embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        if is_query and self._query_instruction:
            texts = [self._query_instruction + t for t in texts]
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]

        async def run(batch: list[str]) -> list[list[float]]:
            async def call() -> Any:
                async with self._sem:
                    kwargs: dict[str, Any] = {"model": self.model_name, "input": batch}
                    if self._send_dimensions:
                        kwargs["dimensions"] = self._dims
                    try:
                        return await self._client.embeddings.create(**kwargs)
                    except openai.BadRequestError as exc:
                        if "dimensions" not in kwargs or "dimension" not in str(exc).lower():
                            raise
                        self._send_dimensions = False
                        kwargs.pop("dimensions")
                        return await self._client.embeddings.create(**kwargs)

            resp = await with_retries(call)
            return [self._fit(d.embedding) for d in sorted(resp.data, key=lambda d: d.index)]

        results = await asyncio.gather(*(run(b) for b in batches))
        return [vec for batch in results for vec in batch]


def make_client(api_key: str | None, base_url: str, timeout_s: float) -> AsyncOpenAI:
    # max_retries=0: retries are handled by `with_retries` with a stricter policy.
    # Local servers (Ollama, vLLM, LM Studio) need no key, but the SDK requires a value.
    return AsyncOpenAI(
        api_key=api_key or "not-needed", base_url=base_url, timeout=timeout_s, max_retries=0
    )
