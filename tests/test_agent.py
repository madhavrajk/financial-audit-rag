"""Agent loop behaviour with a scripted LLM: routing, guard rails, repair, cache, follow-ups."""

from __future__ import annotations

from app.adapters.memory import MemoryAnswerCache
from app.core.agent import Agent
from app.tools import ToolBox
from tests.conftest import FakeLLM, calls, say, tool

REV_SQL = "SELECT revenue FROM income_statements WHERE company_ticker='AAPL' AND fiscal_year=2025"
GOOD = (
    "Apple's net sales in fiscal year 2025 were $416.2 billion.\nSources: [SQL: income_statements]"
)


def make_agent(store, index, script, **kw):
    llm = FakeLLM(script)
    return Agent(llm, ToolBox(store, index), store.schema_description(), **kw), llm


async def test_sql_route_and_grounded_answer(store, index):
    agent, llm = make_agent(store, index, [calls(tool("query_database", sql=REV_SQL)), say(GOOD)])
    r = await agent.run("How much did Apple sell in fiscal 2025?")
    assert r.answer == GOOD
    assert r.tools_used == ["query_database"]
    assert r.grounding is not None and r.grounding.ok
    assert r.sources[0].label == "SQL: income_statements"  # cited sources first
    # The static system prompt is the first message on every call (prompt-cache friendly).
    assert all(c["messages"][0]["content"] == agent.system_prompt for c in llm.calls)


async def test_parallel_mixed_routing(store, index):
    agent, _ = make_agent(
        store,
        index,
        [
            calls(
                tool(
                    "query_database",
                    "a",
                    sql="SELECT fiscal_year, revenue FROM segment_revenue "
                    "WHERE company_ticker='AAPL' AND segment_name='Greater China'",
                ),
                tool(
                    "search_filings",
                    "b",
                    query="Greater China net sales decreased",
                    ticker="AAPL",
                    fiscal_year=2025,
                ),
                content="Plan: get the numbers from SQL and the explanation from the 10-K.",
            ),
            say(
                "Greater China net sales were $64.4 billion in FY2025, down from $67.0 billion. "
                "Sources: [AAPL FY2025 10-K p.24]"
            ),
        ],
    )
    r = await agent.run("Why did Apple sell less in Greater China last year?")
    assert set(r.tools_used) == {"query_database", "search_filings"}
    assert [t.kind for t in r.trace][:3] == ["plan", "tool", "tool"]
    assert r.grounding is not None and r.grounding.ok


async def test_step_budget_forces_final_answer(store, index):
    loop = [calls(tool("query_database", f"c{i}", sql=f"SELECT {i}")) for i in range(3)]
    agent, llm = make_agent(store, index, [*loop, say("Best effort answer.")], max_steps=3)
    r = await agent.run("q")
    assert r.answer == "Best effort answer."
    assert llm.calls[-1]["tool_choice"] == "none"
    assert r.trace[-1].kind == "forced"


async def test_repeated_identical_call_is_short_circuited(store, index):
    same = tool("query_database", sql=REV_SQL)
    agent, _llm = make_agent(store, index, [calls(same), calls(same), say(GOOD)])
    r = await agent.run("q")
    repeated = [t for t in r.trace if t.kind == "tool"][1]
    assert repeated.error and "already made this exact call" in repeated.output
    assert r.answer == GOOD


async def test_bad_tool_calls_go_back_to_model(store, index):
    agent, llm = make_agent(
        store,
        index,
        [
            calls(tool("query_database", sql="DROP TABLE companies")),
            calls(tool("query_database", "c2", sql=REV_SQL)),
            say(GOOD),
        ],
    )
    r = await agent.run("q")
    first = next(t for t in r.trace if t.kind == "tool")
    assert first.error and "read-only" in first.output
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert "read-only" in tool_msgs[0]["content"]
    assert r.answer == GOOD


async def test_ungrounded_answer_triggers_repair(store, index):
    agent, llm = make_agent(
        store,
        index,
        [
            calls(tool("query_database", sql=REV_SQL)),
            say("Apple's revenue was $420.0 billion."),  # wrong
            say(GOOD),  # after repair prompt
        ],
        escalation_model="strong-model",
    )
    r = await agent.run("q")
    assert r.answer == GOOD and r.grounding is not None and r.grounding.ok
    assert "repair" in [t.kind for t in r.trace]
    assert "do not appear in any tool result" in llm.calls[2]["messages"][-1]["content"]


async def test_escalates_when_repair_fails(store, index):
    agent, llm = make_agent(
        store,
        index,
        [
            calls(tool("query_database", sql=REV_SQL)),
            say("Revenue was $420.0 billion."),
            say("Revenue was $421.0 billion."),  # repair still wrong
            say(GOOD),  # escalation model
        ],
        escalation_model="strong-model",
    )
    r = await agent.run("q")
    assert r.answer == GOOD
    assert llm.calls[-1]["model"] == "strong-model"
    assert [t.kind for t in r.trace if t.kind in ("repair", "escalate")] == ["repair", "escalate"]


async def test_unresolved_numbers_are_reported_not_hidden(store, index):
    agent, _ = make_agent(
        store, index, [say("Revenue was $999.0 billion."), say("Revenue was $999.0 billion.")]
    )
    r = await agent.run("q")
    assert r.grounding is not None and r.grounding.unsupported == ["$999.0 billion"]


async def test_follow_up_is_rewritten_and_cache_keyed_on_standalone(store, index):
    cache = MemoryAnswerCache(max_entries=10, ttl_s=60)
    key = lambda q: q.lower()  # noqa: E731
    msft = "Microsoft's total revenue in fiscal year 2025 was $281.7 billion."
    agent, _llm = make_agent(
        store,
        index,
        [
            say("What was Microsoft's total revenue in fiscal year 2025?"),  # rewrite
            calls(
                tool(
                    "query_database",
                    sql="SELECT revenue FROM income_statements WHERE "
                    "company_ticker='MSFT' AND fiscal_year=2025",
                )
            ),
            say(msft),
        ],
    )
    history = [
        {"role": "user", "content": "What was Apple's revenue in FY2025?"},
        {"role": "assistant", "content": GOOD},
    ]
    # Pre-seed the cache with Apple's answer under the *raw* follow-up text: it must not hit.
    cache.set(key("what about microsoft?"), {"answer": GOOD, "standalone_question": "x"})
    r = await agent.run("what about microsoft?", history=history, cache=cache, cache_key=key)
    assert r.answer == msft and not r.cached
    assert r.standalone_question == "What was Microsoft's total revenue in fiscal year 2025?"
    assert r.trace[0].kind == "rewrite"
    # Asking the standalone question again is now a cache hit with zero LLM calls.
    agent2, llm2 = make_agent(store, index, [])
    r2 = await agent2.run(
        "What was Microsoft's total revenue in fiscal year 2025?", cache=cache, cache_key=key
    )
    assert r2.cached and r2.answer == msft and llm2.calls == []


async def test_empty_reply_then_fallback(store, index):
    agent, _ = make_agent(store, index, [say(""), say(""), say("")])
    r = await agent.run("q")
    assert "could not produce a reliable answer" in r.answer


async def test_mistyped_calculator_input_gets_flagged_in_loop(store, index):
    agent, llm = make_agent(
        store,
        index,
        [
            calls(tool("query_database", sql=REV_SQL)),
            calls(tool("calculate", "c2", expressions={"x": "416116 / 1000"})),  # typo
            say("Apple's revenue was $416.2 billion."),
        ],
    )
    await agent.run("q")
    calc_msg = [m for m in llm.calls[2]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "WARNING" in calc_msg and "416116" in calc_msg


def test_clean_answer():
    from app.core.agent import clean_answer

    raw = "Apple's revenue was **$416.2 billion**.【SQL: income_statements】"
    assert clean_answer(raw) == "Apple's revenue was $416.2 billion.[SQL: income_statements]"


async def test_token_budget_forces_final_answer(store, index):
    # FakeLLM reports 100 prompt tokens per call; a budget of 150 trips after the 2nd call.
    loop = [calls(tool("query_database", f"c{i}", sql=f"SELECT {i}")) for i in range(5)]
    agent, llm = make_agent(
        store, index, [*loop, say("Answer from evidence so far.")], max_request_tokens=150
    )
    r = await agent.run("q")
    assert r.trace[-1].kind == "forced" and llm.calls[-1]["tool_choice"] == "none"
    assert len(llm.calls) == 3


async def test_a_printed_page_number_is_cited_as_the_page_read(store, index):
    # The PDF page labelled p.38 prints the 10-K's own page number, 39.
    read = tool("read_filing_pages", document="MSFT FY2025 10-K", start_page=38, end_page=38)
    answer = "Azure and other cloud services revenue grew 34%. Sources: [MSFT FY2025 10-K p.39]"
    agent, _ = make_agent(store, index, [calls(read), say(answer)])
    r = await agent.run("How fast did Azure grow in fiscal 2025?")
    assert r.answer.endswith("[MSFT FY2025 10-K p.38]")
    assert r.trace[-1].kind == "citations" and "p.39 -> p.38" in r.trace[-1].output
    assert r.sources[0].label == "MSFT FY2025 10-K p.38"
