"""Tools the agent can call, wired to the ports.

Each tool's arguments are a Pydantic model. The JSON schema sent to the model is generated
from that model, with the companies, years and documents actually present added as enums,
so the schema, the validation and the data can never drift apart. Invalid arguments, unknown
tools, and recoverable failures come back to the model as tool messages it can correct; they
are never raised to the user.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.core.errors import ToolError
from app.core.ports import DocumentInfo, FinancialStore, Page, Retriever, ToolCall
from app.tools.calc import calculate_many

# Names people use that differ from the registrant name in the data.
EXTRA_ALIASES = {"GOOGLE": "GOOGL", "GOOG": "GOOGL"}
_LEGAL_SUFFIX = re.compile(r"\b(INC|CORP|CORPORATION|CO|COM|COMPANY|LTD|PLC|HOLDINGS|GROUP)\b")
_CITED_PAGE = re.compile(r"\s+p\.?\s*\d+\s*$", re.I)  # "AAPL FY2025 10-K p.23" -> the document
MAX_PAGE_CHARS = 7000
_SQL_TABLE = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", re.I)
MAX_READ_PAGES = 3


def _name_key(name: str) -> str:
    return " ".join(re.sub(r"[^A-Z0-9& ]", " ", name.upper()).split())


class Source(BaseModel):
    kind: Literal["sql", "filing"]
    label: str  # "AAPL FY2025 10-K p.23" or "SQL: income_statements"
    detail: str = ""  # section name, or the SQL text


class ToolOutcome(BaseModel):
    content: str
    sources: list[Source] = Field(default_factory=list)
    error: bool = False
    degraded: bool = False


class QueryDatabaseArgs(BaseModel):
    """Run one read-only SQLite SELECT against the structured financials database."""

    sql: str = Field(description="A single SQLite SELECT statement.")


class SearchFilingsArgs(BaseModel):
    """Hybrid (keyword + semantic) search over filing pages. Returns full page text."""

    query: str = Field(description="What to look for, phrased like the text you expect to find.")
    ticker: str | None = Field(
        default=None, description="Restrict to one company. Strongly recommended."
    )
    fiscal_year: int | None = Field(
        default=None, description="Restrict to one fiscal year's documents."
    )
    form: str | None = Field(default=None, description="Restrict to one document type.")
    section: str | None = Field(
        default=None,
        description="Optional 10-K item code to restrict to, e.g. '1A' (Risk Factors), "
        "'7' (MD&A), '8' (Financial Statements and notes), '1' (Business).",
    )
    k: int = Field(default=3, ge=1, le=6, description="Number of pages to return.")


class ReadFilingPagesArgs(BaseModel):
    """Read specific pages of one document (e.g. the page after a search hit, where a table
    or paragraph continues)."""

    document: str = Field(description='The document as cited, e.g. "AAPL FY2025 10-K".')
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1, description=f"Inclusive; at most {MAX_READ_PAGES} pages per call.")


class CalculateArgs(BaseModel):
    """Evaluate arithmetic exactly. Use for every growth rate, ratio, share, sum or
    difference. Pass several labelled expressions at once."""

    expressions: dict[str, str] = Field(
        description='Label -> arithmetic expression, e.g. {"aapl_growth_pct": '
        '"(416161 - 391035) / 391035 * 100"}. Numbers, + - * / ** %, parentheses, '
        "abs/round/min/max only."
    )


_ARG_MODELS: dict[str, type[BaseModel]] = {
    "query_database": QueryDatabaseArgs,
    "search_filings": SearchFilingsArgs,
    "read_filing_pages": ReadFilingPagesArgs,
    "calculate": CalculateArgs,
}


def _page_block(p: Page) -> str:
    text = p.text if len(p.text) <= MAX_PAGE_CHARS else p.text[:MAX_PAGE_CHARS] + "\n[...truncated]"
    return f"=== [{p.citation}] {p.section}\n{text}"


def _format_rows(columns: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        return "NULL" if v is None else str(v)

    lines = [" | ".join(columns)] + [" | ".join(fmt(v) for v in r) for r in rows]
    return "\n".join(lines)


class ToolBox:
    """The tools over one corpus. Without a database (or with an empty one), query_database
    is not offered and every answer comes from the filings."""

    def __init__(self, store: FinancialStore | None, retriever: Retriever) -> None:
        self._retriever = retriever
        self.documents: list[DocumentInfo] = retriever.catalog()
        self.db_companies = store.companies() if store is not None else {}
        self._store = store if self.db_companies else None
        names = {d.ticker: d.company for d in self.documents if d.company}
        names.update({t: name for t, (name, _) in self.db_companies.items()})
        self.company_names = dict(sorted(names.items()))
        self.tickers = sorted({d.ticker for d in self.documents} | set(self.db_companies))
        self._aliases = {t: t for t in self.tickers}
        for t, name in self.company_names.items():
            self._aliases.setdefault(_name_key(name), t)
            self._aliases.setdefault(" ".join(_LEGAL_SUFFIX.sub(" ", _name_key(name)).split()), t)
        for alias, t in EXTRA_ALIASES.items():
            if t in self.tickers:
                self._aliases.setdefault(alias, t)
        self._doc_names = {_name_key(d.name): d.name for d in self.documents}
        self._models = {
            n: m for n, m in _ARG_MODELS.items() if n != "query_database" or self._store
        }

    def resolve_ticker(self, value: str | None) -> str | None:
        if not value:
            return None
        ticker = self._aliases.get(_name_key(value))
        if ticker is None:
            raise ToolError(
                f"Unknown company {value!r}. Companies available: {', '.join(self.tickers)}."
            )
        return ticker

    def specs(self) -> list[dict[str, Any]]:
        """OpenAI-format tool specs, in a fixed order (keeps the prompt prefix cacheable)."""
        enums: dict[str, list[Any]] = {
            "ticker": self.tickers,
            "fiscal_year": sorted({d.fiscal_year for d in self.documents}),
            "form": sorted({d.form for d in self.documents}),
            "document": [d.name for d in self.documents],
        }
        specs = []
        for name, model in self._models.items():
            schema = model.model_json_schema()
            schema.pop("title", None)
            schema.pop("description", None)  # already in the function description
            props = schema.get("properties", {})
            if len(enums["form"]) <= 1:
                props.pop("form", None)  # one document type: nothing to filter
            for field, prop in props.items():
                prop.pop("title", None)
                if enums.get(field):
                    branches = prop.get("anyOf", [prop])
                    target = next(b for b in branches if b.get("type") != "null")
                    target["enum"] = enums[field]
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (model.__doc__ or "").strip().replace("\n    ", " "),
                        "parameters": schema,
                    },
                }
            )
        return specs

    async def execute(self, call: ToolCall) -> ToolOutcome:
        model = self._models.get(call.name)
        if model is None:
            return ToolOutcome(
                error=True,
                content=(f"Unknown tool {call.name!r}. Available: {', '.join(self._models)}."),
            )
        try:
            raw = json.loads(call.arguments or "{}")
            args = model.model_validate(raw)
        except json.JSONDecodeError as exc:
            return ToolOutcome(error=True, content=f"Arguments are not valid JSON: {exc.msg}")
        except ValidationError as exc:
            errs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return ToolOutcome(error=True, content=f"Invalid arguments for {call.name}: {errs}")
        try:
            if isinstance(args, QueryDatabaseArgs):
                return await self._query_database(args)
            if isinstance(args, SearchFilingsArgs):
                return await self._search(args)
            if isinstance(args, ReadFilingPagesArgs):
                return self._read_pages(args)
            if isinstance(args, CalculateArgs):
                return ToolOutcome(content=json.dumps(calculate_many(args.expressions)))
        except ToolError as exc:
            return ToolOutcome(error=True, content=str(exc))
        raise AssertionError(f"unhandled tool {call.name}")  # pragma: no cover

    async def _query_database(self, args: QueryDatabaseArgs) -> ToolOutcome:
        assert self._store is not None  # the tool is only offered with a database
        result = await asyncio.to_thread(self._store.query, args.sql)
        body = _format_rows(result.columns, result.rows) if result.rows else "(no rows)"
        if result.truncated:
            body += f"\n[truncated to {len(result.rows)} rows; aggregate or filter instead]"
        tables = list(dict.fromkeys(t.lower() for t in _SQL_TABLE.findall(args.sql)))
        return ToolOutcome(
            content=body,
            sources=[Source(kind="sql", label=f"SQL: {t}", detail=args.sql.strip()) for t in tables]
            or [Source(kind="sql", label="SQL: financials.db", detail=args.sql.strip())],
        )

    async def _search(self, args: SearchFilingsArgs) -> ToolOutcome:
        res = await self._retriever.search(
            args.query,
            ticker=self.resolve_ticker(args.ticker),
            fiscal_year=args.fiscal_year,
            form=args.form,
            section=args.section,
            k=args.k,
        )
        if not res.hits:
            return ToolOutcome(
                content=res.note or "No relevant pages found.", degraded=res.degraded
            )
        blocks = [_page_block(h.page) for h in res.hits]
        if res.degraded:
            blocks.insert(0, "[note: semantic search unavailable; keyword-only results]")
        return ToolOutcome(
            content="\n\n".join(blocks),
            sources=[
                Source(kind="filing", label=h.page.citation, detail=h.page.section)
                for h in res.hits
            ],
            degraded=res.degraded,
        )

    def _read_pages(self, args: ReadFilingPagesArgs) -> ToolOutcome:
        if args.end_page < args.start_page:
            raise ToolError("end_page must be >= start_page")
        document = self._doc_names.get(_name_key(_CITED_PAGE.sub("", args.document)))
        if document is None:
            listed = ", ".join(d.name for d in self.documents)
            raise ToolError(f"Unknown document {args.document!r}. Documents: {listed}.")
        end = min(args.end_page, args.start_page + MAX_READ_PAGES - 1)
        pages = self._retriever.get_pages(document, args.start_page, end)
        if not pages:
            raise ToolError(f"No pages {args.start_page}-{end} in {document}.")
        return ToolOutcome(
            content="\n\n".join(_page_block(p) for p in pages),
            sources=[Source(kind="filing", label=p.citation, detail=p.section) for p in pages],
        )
