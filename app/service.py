"""Application service: the single entry point shared by the HTTP API, the CLI, and evals.

It owns request-level concerns (input validation, cache key, deadline, tracing, versioning),
so every interface behaves the same.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.core.agent import Agent, AgentResult, EventCallback
from app.core.errors import UpstreamError, UserInputError
from app.core.ports import AnswerCache, Message, TraceSink
from app.core.prompts import PROMPT_VERSION

log = logging.getLogger(__name__)
_WS = re.compile(r"\s+")


class Versions(BaseModel):
    prompt: str
    index: str
    model: str


class ChatResult(BaseModel):
    """What every interface returns."""

    answer: str
    request_id: str
    standalone_question: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    grounding: dict[str, Any] | None = None
    tools_used: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False
    degraded: bool = False
    models: list[str] = Field(default_factory=list)
    versions: Versions


def normalize_question(q: str) -> str:
    return _WS.sub(" ", q.strip().lower()).rstrip("?.! ")


class ChatService:
    def __init__(
        self,
        agent: Agent,
        cache: AnswerCache,
        traces: TraceSink,
        *,
        model: str,
        index_version: str,
        max_question_chars: int,
        timeout_s: float,
    ) -> None:
        self._agent = agent
        self._cache = cache
        self._traces = traces
        self._versions = Versions(prompt=PROMPT_VERSION, index=index_version, model=model)
        self._max_chars = max_question_chars
        self._timeout_s = timeout_s

    @property
    def versions(self) -> Versions:
        return self._versions

    def _cache_key(self, standalone: str) -> str:
        raw = "|".join(
            [
                normalize_question(standalone),
                self._versions.model,
                self._versions.index,
                self._versions.prompt,
            ]
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def validate(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            raise UserInputError("Question is empty.")
        if len(q) > self._max_chars:
            raise UserInputError(f"Question exceeds {self._max_chars} characters.", 413)
        return q

    async def ask(
        self,
        question: str,
        *,
        history: list[Message] | None = None,
        session_id: str | None = None,
        use_cache: bool = True,
        on_event: EventCallback | None = None,
    ) -> ChatResult:
        q = self.validate(question)
        request_id = uuid.uuid4().hex[:12]
        t0 = time.monotonic()
        record: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
            "question": q,
            "history_turns": len(history or []),
            "versions": self._versions.model_dump(),
        }
        try:
            result: AgentResult = await asyncio.wait_for(
                self._agent.run(
                    q,
                    history=history,
                    user=session_id or request_id,
                    on_event=on_event,
                    cache=self._cache if use_cache else None,
                    cache_key=self._cache_key,
                ),
                timeout=self._timeout_s,
            )
        except TimeoutError:
            record.update(error="timeout", latency_ms=int((time.monotonic() - t0) * 1000))
            self._traces.emit(record)
            raise UpstreamError(
                f"Request exceeded {self._timeout_s:.0f}s.", retry_after_s=5
            ) from None
        except Exception as exc:
            record.update(
                error=f"{type(exc).__name__}: {exc}", latency_ms=int((time.monotonic() - t0) * 1000)
            )
            self._traces.emit(record)
            raise

        latency_ms = int((time.monotonic() - t0) * 1000)
        out = ChatResult(
            answer=result.answer,
            request_id=request_id,
            standalone_question=result.standalone_question,
            sources=[s.model_dump() for s in result.sources],
            trace=[t.model_dump() for t in result.trace],
            grounding=(
                {"checked": result.grounding.checked, "unsupported": result.grounding.unsupported}
                if result.grounding
                else None
            ),
            tools_used=result.tools_used,
            usage=result.usage.model_dump(),
            cost_usd=result.cost_usd,
            latency_ms=latency_ms,
            cached=result.cached,
            degraded=result.degraded,
            models=result.models,
            versions=self._versions,
        )
        record.update(out.model_dump(exclude={"trace"}), trace_steps=len(result.trace))
        self._traces.emit(record)
        log.info(
            "request %s: %dms, %d llm calls, $%.5f, cached=%s, tools=%s",
            request_id,
            latency_ms,
            result.llm_calls,
            result.cost_usd,
            result.cached,
            result.tools_used,
        )
        return out
