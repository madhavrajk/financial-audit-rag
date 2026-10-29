"""HTTP API and UI.

POST /api/chat   {"question": "...", "history"?: [...], "session_id"?: "..."}
                 → JSON {"answer": "...", sources, trace, grounding, usage, cost_usd, ...}
                 → or SSE (Accept: text/event-stream): progress events, then an `answer`
                   event with data {"content": "..."}, then a `done` event with metadata.
GET  /           chat UI
GET  /healthz    liveness (process is up)
GET  /readyz     readiness (DB, index and provider settings); 503 with reasons otherwise
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.container import Container, build_container
from app.core.config import Settings, get_settings
from app.core.errors import NotReadyError, UpstreamAuthError, UpstreamError, UserInputError
from app.service import ChatResult

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    session_id: str | None = Field(default=None, max_length=100)


class RateLimiter:
    """Per-client token bucket. In-process is fine for one replica; use Redis for many."""

    def __init__(self, per_minute: int) -> None:
        self._rate = per_minute / 60.0
        self._cap = float(per_minute)
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        tokens, last = self._buckets.get(client, (self._cap, now))
        tokens = min(self._cap, tokens + (now - last) * self._rate)
        if tokens < 1:
            self._buckets[client] = (tokens, now)
            return False
        self._buckets[client] = (tokens - 1, now)
        return True


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container or build_container(settings)
        app.state.container = c
        if c.problems:
            for p in c.problems:
                log.error("NOT READY: %s", p)
        else:
            log.info(
                "ready: index %s, model %s",
                c.index.index_version if c.index else "-",
                settings.llm_model,
            )
        yield

    app = FastAPI(title="10-K Research Assistant", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[f"http://localhost:{settings.port}", f"http://127.0.0.1:{settings.port}"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )
    limiter = RateLimiter(settings.rate_limit_per_minute)

    def error(status: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return error(400, f"Invalid request: {loc} {first.get('msg', '')}".strip())

    def to_response(exc: Exception) -> JSONResponse:
        if isinstance(exc, UserInputError):
            return error(exc.status_code, str(exc))
        if isinstance(exc, UpstreamAuthError):
            return error(503, str(exc))
        if isinstance(exc, UpstreamError):
            return error(503, str(exc), {"Retry-After": str(exc.retry_after_s or 10)})
        if isinstance(exc, NotReadyError):
            return error(503, str(exc))
        log.exception("unhandled error", exc_info=exc)
        return error(500, "Internal error; see server logs.")

    def service_or_503(request: Request) -> Any:
        c: Container = request.app.state.container
        if not c.ready or c.service is None:
            raise NotReadyError("Service not ready: " + "; ".join(c.problems))
        return c.service

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        c: Container = request.app.state.container
        body: dict[str, Any] = {"ready": c.ready, "problems": c.problems}
        if c.index is not None:
            docs = c.index.catalog()
            years = [d.fiscal_year for d in docs]
            body.update(
                index_version=c.index.index_version,
                dense_search=c.index.dense_enabled,
                companies=sorted({d.ticker for d in docs}),
                fiscal_years=[min(years), max(years)] if years else [],
                documents=c.index.available_docs(),
            )
        body["model"] = settings.llm_model
        return JSONResponse(body, status_code=200 if c.ready else 503)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/api/chat", response_model=None)
    async def chat(req: ChatRequest, request: Request) -> JSONResponse | StreamingResponse:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            return error(429, "Rate limit exceeded; slow down.", {"Retry-After": "5"})
        try:
            service = service_or_503(request)
            service.validate(req.question)
        except Exception as exc:
            return to_response(exc)
        history = [t.model_dump() for t in req.history]
        if "text/event-stream" in request.headers.get("accept", ""):
            return StreamingResponse(
                _sse(service, req, history, request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        task = asyncio.create_task(
            service.ask(req.question, history=history, session_id=req.session_id)
        )
        try:
            # Cancel the in-flight LLM work if the client goes away (saves tokens).
            while not task.done():
                await asyncio.wait({task}, timeout=0.5)
                if not task.done() and await request.is_disconnected():
                    task.cancel()
                    return error(499, "client disconnected")
            result: ChatResult = task.result()
        except Exception as exc:
            return to_response(exc)
        return JSONResponse(result.model_dump())

    return app


def _sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _sse(
    service: Any, req: ChatRequest, history: list[dict[str, str]], request: Request
) -> AsyncIterator[str]:
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def on_event(kind: str, data: dict[str, Any]) -> None:
        if kind != "answer":  # the final answer is sent once, below, with metadata
            await queue.put(("status", {"kind": kind, **data}))

    task = asyncio.create_task(
        service.ask(req.question, history=history, session_id=req.session_id, on_event=on_event)
    )
    try:
        while not task.done() or not queue.empty():
            try:
                kind, data = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield _sse_event(kind, data)
            except TimeoutError:
                if await request.is_disconnected():
                    task.cancel()
                    return
        try:
            result: ChatResult = task.result()
        except Exception as exc:
            status = 503 if isinstance(exc, UpstreamError | NotReadyError) else 500
            msg = str(exc) if status == 503 else "Internal error; see server logs."
            yield _sse_event("error", {"status": status, "error": msg})
            return
        yield _sse_event("answer", {"content": result.answer})
        yield _sse_event("done", result.model_dump(exclude={"answer"}))
    finally:
        if not task.done():
            task.cancel()


app = create_app()


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    s = get_settings()
    uvicorn.run("app.api.server:app", host=s.host, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
