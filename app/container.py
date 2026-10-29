"""Composition root: the one place that picks concrete adapters for each port.

Everything else depends on interfaces. A production deployment changes this file (or
config) to wire Postgres / Qdrant / Redis adapters; the agent, tools and API are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.adapters.local_index import LocalPageIndex
from app.adapters.memory import JsonlTraceSink, MemoryAnswerCache, NullAnswerCache
from app.adapters.openai_compat import OpenAICompatEmbedder, OpenAICompatLLM, make_client
from app.adapters.sqlite_store import SQLiteFinancialStore
from app.core.agent import Agent
from app.core.config import Settings
from app.core.errors import NotReadyError
from app.core.ports import AnswerCache, LLMClient, model_key
from app.core.pricing import PRICES
from app.service import ChatService
from app.tools import ToolBox

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    service: ChatService | None = None
    index: LocalPageIndex | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.service is not None and not self.problems


def build_container(settings: Settings, *, llm: LLMClient | None = None) -> Container:
    """Build everything. Missing resources are recorded as `problems` (served by /readyz and
    shown as a clear startup message) instead of crashing with a stack trace."""
    c = Container(settings=settings)
    try:
        store = SQLiteFinancialStore(settings.db_path)
    except FileNotFoundError as exc:
        c.problems.append(f"{exc}. See data/README.md.")
        return c

    PRICES.update({model_key(m): p for m, p in settings.model_prices.items()})
    client = None
    embedder = None
    provider_problems = settings.provider_problems()
    if not provider_problems:
        client = make_client(settings.api_key, settings.llm_base_url, settings.llm_timeout_s)
    elif llm is None:
        c.problems.extend(provider_problems)
    if settings.embed_model and (settings.embed_key or settings.is_local(settings.embed_url)):
        embed_client = (
            client
            if client is not None and settings.embed_url == settings.llm_base_url
            else make_client(settings.embed_key, settings.embed_url, settings.llm_timeout_s)
        )
        embedder = OpenAICompatEmbedder(
            embed_client, model=settings.embed_model, dims=settings.embed_dims
        )

    try:
        c.index = LocalPageIndex(settings.index_dir, embedder)
    except NotReadyError as exc:
        c.problems.append(str(exc))
        return c
    if not c.index.dense_enabled:
        log.warning("dense search disabled; using BM25 only (degraded)")

    if llm is None:
        if client is None:
            return c
        llm = OpenAICompatLLM(
            client,
            default_model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            reasoning_effort=settings.reasoning_effort or None,
            max_concurrency=settings.max_concurrent_llm_calls,
        )

    agent = Agent(
        llm,
        ToolBox(store, c.index),
        store.schema_description(),
        max_steps=settings.max_steps,
        escalation_model=settings.escalation_model,
        rewrite_model=settings.rewrite_model,
        max_request_tokens=settings.max_request_tokens,
    )
    cache: AnswerCache = (
        MemoryAnswerCache(
            max_entries=settings.answer_cache_max_entries, ttl_s=settings.answer_cache_ttl_s
        )
        if settings.answer_cache == "memory"
        else NullAnswerCache()
    )
    c.service = ChatService(
        agent,
        cache,
        JsonlTraceSink(settings.trace_dir / "requests.jsonl"),
        model=settings.llm_model,
        index_version=c.index.index_version,
        max_question_chars=settings.max_question_chars,
        timeout_s=settings.request_timeout_s,
    )
    return c
