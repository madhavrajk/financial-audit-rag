from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.adapters.local_index import LocalPageIndex
from app.adapters.sqlite_store import SQLiteFinancialStore
from app.core.config import PROJECT_ROOT
from app.core.ports import LLMResponse, Message, ToolCall, ToolChoice, Usage
from app.ingest.build_index import build_index, discover

DB = PROJECT_ROOT / "data" / "financials.db"
PDFS = PROJECT_ROOT / "data" / "pdfs"


class FakeLLM:
    """Scripted LLM: returns queued responses in order and records every request."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: Any = None,
        tool_choice: ToolChoice = "auto",
        model: str | None = None,
        max_tokens: int | None = None,
        user: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tool_choice": tool_choice,
                "model": model,
                "tools": tools,
            }
        )
        if not self.script:
            return LLMResponse(content="(script exhausted)", model=model or "fake")
        resp = self.script.pop(0)
        return resp.model_copy(
            update={
                "model": model or "fake",
                "usage": Usage(prompt_tokens=100, completion_tokens=10),
            }
        )


def tool(name: str, call_id: str = "c1", **args: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(args))


def calls(*tcs: ToolCall, content: str | None = None) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=list(tcs))


def say(text: str) -> LLMResponse:
    return LLMResponse(content=text)


@pytest.fixture(scope="session")
def store() -> SQLiteFinancialStore:
    return SQLiteFinancialStore(DB)


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A BM25-only index built from the real PDFs (no API calls)."""
    d = tmp_path_factory.mktemp("index") / "index"
    asyncio.run(build_index(discover(PDFS), d, embedder=None, dims=1024))
    return d


@pytest.fixture(scope="session")
def index(index_dir: Path) -> LocalPageIndex:
    return LocalPageIndex(index_dir, embedder=None)
