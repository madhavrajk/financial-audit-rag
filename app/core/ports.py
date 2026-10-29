"""Ports (interfaces) the agent depends on, plus the small domain types they exchange.

The agent and tools only ever see these Protocols. Concrete backends live in app/adapters/
and are wired together in app/container.py. Swapping SQLite for Postgres, numpy for Qdrant,
or the in-process cache for Redis is a new adapter, not a change to the agent.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # raw JSON string exactly as the model produced it
    # Provider fields to send back unchanged on the next turn. Gemini 3 attaches thought
    # signatures here and rejects a follow-up request that drops them.
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    cost_usd: float = 0.0


class Page(BaseModel):
    """One filing page: the retrieval unit. Pages keep tables intact and make citations exact."""

    doc_id: str  # e.g. "AAPL_FY2025"
    ticker: str
    fiscal_year: int
    company: str = ""  # registrant name, when the source says so
    form: str = "10-K"  # 10-K, 10-Q, 8-K, Earnings release, ...
    period: str = ""  # label when not a whole fiscal year, e.g. "FY2023 Q2"; default FY{year}
    page: int  # 1-based page number (what a PDF viewer shows, or the filing's page breaks)
    section: str  # display label, e.g. "Item 7. MD&A"
    items: list[str] = Field(default_factory=list)  # 10-K item codes on this page, e.g. ["6", "7"]
    text: str
    sha256: str

    @property
    def document(self) -> str:
        """The document's name in citations and in read_filing_pages: "AAPL FY2025 10-K"."""
        return f"{self.ticker} {self.period or f'FY{self.fiscal_year}'} {self.form}"

    @property
    def citation(self) -> str:
        return f"{self.document} p.{self.page}"

    @property
    def header(self) -> str:
        return f"{self.document} · {self.section} · page {self.page}"


class DocumentInfo(BaseModel):
    name: str  # "AAPL FY2025 10-K"
    ticker: str
    company: str
    fiscal_year: int
    form: str
    pages: int


class PageHit(BaseModel):
    page: Page
    score: float


class SearchResult(BaseModel):
    hits: list[PageHit]
    degraded: bool = False  # True when dense search was unavailable and BM25 alone was used
    note: str = ""


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool = False


Message = dict[str, Any]  # OpenAI-format chat message
ToolChoice = Literal["auto", "none", "required"]


def model_key(model: str) -> str:
    """Provider-neutral model name, so the same weights compare equal across providers.

    "org/models/qwen3-embedding-8b", "Qwen/Qwen3-Embedding-8B" and "qwen3-embedding:8b"
    all map to "qwen3-embedding-8b".
    """
    return model.rsplit("/", 1)[-1].lower().replace(":", "-")


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = "auto",
        model: str | None = None,
        max_tokens: int | None = None,
        user: str | None = None,
    ) -> LLMResponse: ...


class Embedder(Protocol):
    model_name: str

    async def embed(self, texts: list[str], *, is_query: bool) -> list[list[float]]: ...


class FinancialStore(Protocol):
    def query(self, sql: str) -> QueryResult: ...

    def schema_description(self) -> str: ...

    def companies(self) -> dict[str, tuple[str, str]]:
        """Ticker -> (company name, fiscal year end month) for companies with data."""
        ...


class Retriever(Protocol):
    index_version: str

    async def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        form: str | None = None,
        section: str | None = None,
        k: int = 4,
    ) -> SearchResult: ...

    def get_pages(self, document: str, start: int, end: int) -> list[Page]: ...

    def catalog(self) -> list[DocumentInfo]: ...


class AnswerCache(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...

    def set(self, key: str, value: dict[str, Any]) -> None: ...


class TraceSink(Protocol):
    def emit(self, record: dict[str, Any]) -> None: ...
