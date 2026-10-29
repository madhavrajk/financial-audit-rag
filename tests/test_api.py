"""HTTP contract: JSON `answer`, SSE `answer` event, error mapping, readiness, rate limit."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.adapters.memory import JsonlTraceSink, MemoryAnswerCache
from app.api.server import create_app
from app.container import Container
from app.core.agent import Agent
from app.core.config import Settings
from app.service import ChatService
from app.tools import ToolBox
from tests.conftest import FakeLLM, calls, say, tool

REV_SQL = "SELECT revenue FROM income_statements WHERE company_ticker='AAPL' AND fiscal_year=2025"
GOOD = "Apple's net sales in fiscal year 2025 were $416.2 billion."


def make_client(store, index, tmp_path, script, **settings_kw) -> TestClient:
    settings = Settings(trace_dir=tmp_path, **settings_kw)
    agent = Agent(FakeLLM(script), ToolBox(store, index), store.schema_description())
    service = ChatService(
        agent,
        MemoryAnswerCache(max_entries=10, ttl_s=60),
        JsonlTraceSink(tmp_path / "t.jsonl"),
        model="fake",
        index_version=index.index_version,
        max_question_chars=4000,
        timeout_s=30,
    )
    container = Container(settings=settings, service=service, index=index)
    return TestClient(create_app(settings, container))


@pytest.fixture
def script():
    return [calls(tool("query_database", sql=REV_SQL)), say(GOOD)]


def test_json_contract(store, index, tmp_path, script):
    with make_client(store, index, tmp_path, script) as c:
        r = c.post("/api/chat", json={"question": "What was Apple's revenue in FY2025?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == GOOD
    assert body["tools_used"] == ["query_database"]
    assert body["versions"]["prompt"] and body["versions"]["index"]
    assert (tmp_path / "t.jsonl").read_text().count("\n") == 1  # one trace record


def test_sse_contract(store, index, tmp_path, script):
    with make_client(store, index, tmp_path, script) as c:
        r = c.post(
            "/api/chat",
            json={"question": "Apple revenue FY2025?"},
            headers={"Accept": "text/event-stream"},
        )
    events = [blk for blk in r.text.split("\n\n") if blk.strip()]
    parsed = [
        (
            b.split("\n")[0].removeprefix("event: "),
            json.loads(b.split("\n")[1].removeprefix("data: ")),
        )
        for b in events
    ]
    names = [n for n, _ in parsed]
    assert "answer" in names and names[-1] == "done"
    assert dict(parsed)["answer"] == {"content": GOOD}


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"question": ""}, 400),
        ({"question": "   "}, 400),
        ({"question": "x" * 5000}, 400),
        ({}, 400),
    ],
)
def test_bad_input(store, index, tmp_path, payload, status):
    with make_client(store, index, tmp_path, []) as c:
        r = c.post("/api/chat", json=payload)
    assert r.status_code == status and "error" in r.json()


def test_not_ready_returns_503(tmp_path):
    settings = Settings(trace_dir=tmp_path)
    container = Container(settings=settings, problems=["LLM_API_KEY is not set"])
    with TestClient(create_app(settings, container)) as c:
        assert c.get("/healthz").status_code == 200
        ready = c.get("/readyz")
        assert ready.status_code == 503 and ready.json()["problems"]
        r = c.post("/api/chat", json={"question": "hi"})
        assert r.status_code == 503 and "LLM_API_KEY" in r.json()["error"]


def test_rate_limit(store, index, tmp_path):
    script = [say("ok")] * 10
    with make_client(store, index, tmp_path, script, rate_limit_per_minute=2) as c:
        codes = [c.post("/api/chat", json={"question": f"q{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_ui_is_served(store, index, tmp_path):
    with make_client(store, index, tmp_path, []) as c:
        r = c.get("/")
    assert r.status_code == 200 and "textContent" in r.text and ".innerHTML =" not in r.text


def test_trace_write_failure_does_not_fail_request(store, index, tmp_path, script):
    blocked = tmp_path / "ro"
    blocked.mkdir()
    sink = JsonlTraceSink(blocked / "t.jsonl")
    blocked.chmod(0o500)  # read-only directory
    try:
        sink.emit({"x": 1})  # must not raise
    finally:
        blocked.chmod(0o700)
