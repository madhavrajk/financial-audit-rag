"""The agent: one tool-calling loop (ReAct-style, using native function calling).

Why a single loop rather than a router or a planner/executor graph: questions range from
one SQL lookup to "compute growth from SQL, then find management's explanation in the
10-K". A fixed router is brittle on mixed questions, and a separate planner adds latency
and another failure point. The system prompt asks for a short plan on multi-part questions,
which gives most of the planning benefit inside the same loop.

Guard rails, each mapped to a failure mode:
  * step + token budgets, then a forced answer with tools disabled → runaway loops / cost
  * repeated identical tool call is short-circuited                → loops on a bad query
  * invalid/unknown tool calls go back to the model as errors      → self-correction
  * calculator inputs checked against evidence (in-loop warning)   → mistyped numbers
  * numeric grounding check, then 1 repair round, then escalation  → invented numbers
  * parallel tool calls run concurrently (asyncio.gather)          → latency
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.grounding import (
    GroundingReport,
    check_grounding,
    fix_citations,
    unsupported_operands,
)
from app.core.ports import AnswerCache, LLMClient, LLMResponse, Message, ToolCall, Usage
from app.core.prompts import FORCE_ANSWER, REPAIR_TEMPLATE, REWRITE_SYSTEM, build_system_prompt
from app.tools import Source, ToolBox, ToolOutcome

log = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
MAX_PARALLEL_CALLS = 8
REPAIR_STEPS = 3
FALLBACK_ANSWER = (
    "I could not produce a reliable answer from the provided data. "
    "Please rephrase the question or narrow it to one company and fiscal year."
)


class TraceStep(BaseModel):
    step: int
    kind: Literal["rewrite", "plan", "tool", "repair", "escalate", "forced", "citations"]
    name: str | None = None
    args: Any = None
    output: str = ""
    error: bool = False
    latency_ms: int = 0


class AgentResult(BaseModel):
    answer: str
    standalone_question: str
    sources: list[Source] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    grounding: GroundingReport | None = None
    tools_used: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    llm_calls: int = 0
    models: list[str] = Field(default_factory=list)
    degraded: bool = False
    cached: bool = False


class _RunState:
    """Mutable bookkeeping for one request."""

    def __init__(self) -> None:
        self.trace: list[TraceStep] = []
        self.sources: list[Source] = []
        self.evidence: list[str] = []
        self.tools_used: list[str] = []
        self.usage = Usage()
        self.cost = 0.0
        self.llm_calls = 0
        self.models: list[str] = []
        self.degraded = False
        self.seen_calls: dict[str, str] = {}

    def account(self, resp: LLMResponse) -> None:
        self.usage = self.usage + resp.usage
        self.cost += resp.cost_usd
        self.llm_calls += 1
        if resp.model and resp.model not in self.models:
            self.models.append(resp.model)


_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)


def clean_answer(text: str) -> str:
    """Normalise model formatting quirks: 【cite】 → [cite], strip bold markers.

    Done deterministically because some models ignore formatting instructions, and
    citations must use [..] to be matched against sources.
    """
    text = text.replace("【", "[").replace("】", "]")
    return _BOLD.sub(lambda m: m.group(1) or m.group(2), text).strip()


def _preview(text: str, n: int = 600) -> str:
    return text if len(text) <= n else text[:n] + " …"


def _call_signature(call: ToolCall) -> str:
    try:
        args = json.dumps(json.loads(call.arguments or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        args = call.arguments
    return f"{call.name}:{args}"


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolBox,
        schema: str,
        *,
        max_steps: int = 8,
        escalation_model: str | None = None,
        rewrite_model: str | None = None,
        max_request_tokens: int = 120_000,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._tool_specs = tools.specs()
        self._system = build_system_prompt(
            schema, tools.documents, tools.company_names, tools.db_companies
        )
        self._max_steps = max_steps
        self._escalation_model = escalation_model or None
        self._rewrite_model = rewrite_model or None
        self._max_request_tokens = max_request_tokens

    @property
    def system_prompt(self) -> str:
        return self._system

    async def run(
        self,
        question: str,
        *,
        history: list[Message] | None = None,
        user: str | None = None,
        on_event: EventCallback | None = None,
        cache: AnswerCache | None = None,
        cache_key: Callable[[str], str] | None = None,
    ) -> AgentResult:
        st = _RunState()

        async def emit(kind: str, data: dict[str, Any]) -> None:
            if on_event is not None:
                await on_event(kind, data)

        standalone = question
        if history:
            standalone = await self.rewrite(question, history, st)
            await emit("rewrite", {"question": standalone})
        st.evidence.append(standalone)

        # The cache is keyed on the STANDALONE question, so a context-dependent follow-up
        # ("what about Microsoft?") can never be served another conversation's answer.
        key = cache_key(standalone) if cache is not None and cache_key is not None else None
        if cache is not None and key is not None and (hit := cache.get(key)) is not None:
            result = AgentResult.model_validate(hit)
            result.cached = True
            result.usage, result.cost_usd, result.llm_calls = (
                st.usage,
                round(st.cost, 6),
                st.llm_calls,
            )
            result.trace = st.trace + result.trace
            await emit("answer", {"content": result.answer})
            return result

        messages: list[Message] = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": standalone},
        ]
        answer = await self._loop(messages, st, self._max_steps, None, user, emit)

        report = check_grounding(answer, st.evidence)
        if not report.ok:
            # Repair round with the same model, then once more with a stronger model.
            for attempt, model in enumerate([None, self._escalation_model]):
                if attempt == 1 and not model:
                    break
                kind = "repair" if attempt == 0 else "escalate"
                st.trace.append(
                    TraceStep(
                        step=len(st.trace) + 1,
                        kind=kind,
                        output=f"unsupported numbers: {report.unsupported}",
                        args={"model": model} if model else None,
                    )
                )
                await emit(kind, {"unsupported": report.unsupported, "model": model})
                messages.append({"role": "assistant", "content": answer})
                messages.append(
                    {
                        "role": "user",
                        "content": REPAIR_TEMPLATE.format(numbers=", ".join(report.unsupported)),
                    }
                )
                answer = await self._loop(messages, st, REPAIR_STEPS, model, user, emit)
                report = check_grounding(answer, st.evidence)
                if report.ok:
                    break

        answer, moved = fix_citations(answer, st.evidence)
        if moved:
            st.trace.append(
                TraceStep(
                    step=len(st.trace) + 1,
                    kind="citations",
                    output="cited the filing's printed page number: " + "; ".join(moved),
                )
            )
        cited = [s for s in st.sources if s.label in answer]
        uncited = [s for s in st.sources if s.label not in answer]
        result = AgentResult(
            answer=answer,
            standalone_question=standalone,
            sources=cited + uncited,
            trace=st.trace,
            grounding=report,
            tools_used=st.tools_used,
            usage=st.usage,
            cost_usd=round(st.cost, 6),
            llm_calls=st.llm_calls,
            models=st.models,
            degraded=st.degraded,
        )
        # Only cache answers that passed the grounding check on full (non-degraded) search.
        if cache is not None and key is not None and report.ok and not st.degraded:
            cache.set(key, result.model_dump())
        await emit("answer", {"content": answer})
        return result

    async def rewrite(self, question: str, history: list[Message], st: _RunState) -> str:
        t0 = time.monotonic()
        turns = [m for m in history if m.get("role") in ("user", "assistant")][-10:]
        convo = "\n".join(f"{m['role'].upper()}: {str(m.get('content', ''))[:1500]}" for m in turns)
        resp = await self._llm.complete(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Conversation:\n{convo}\n\nLatest message: {question}",
                },
            ],
            model=self._rewrite_model,
            max_tokens=1024,
        )
        st.account(resp)
        standalone = (resp.content or "").strip().strip('"') or question
        st.trace.append(
            TraceStep(
                step=len(st.trace) + 1,
                kind="rewrite",
                output=standalone,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        )
        return standalone

    async def _loop(
        self,
        messages: list[Message],
        st: _RunState,
        max_steps: int,
        model: str | None,
        user: str | None,
        emit: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> str:
        empty_replies = 0
        for _ in range(max_steps):
            t0 = time.monotonic()
            resp = await self._llm.complete(
                messages, tools=self._tool_specs, tool_choice="auto", model=model, user=user
            )
            st.account(resp)
            if resp.tool_calls:
                if resp.content:
                    st.trace.append(
                        TraceStep(
                            step=len(st.trace) + 1,
                            kind="plan",
                            output=_preview(resp.content),
                            latency_ms=int((time.monotonic() - t0) * 1000),
                        )
                    )
                    await emit("plan", {"content": resp.content})
                await self._run_tools(resp, messages, st, emit)
                if st.usage.prompt_tokens > self._max_request_tokens:
                    log.warning(
                        "token budget exceeded (%d); forcing final answer", st.usage.prompt_tokens
                    )
                    break
                continue
            if resp.content and resp.content.strip():
                return clean_answer(resp.content)
            empty_replies += 1
            if empty_replies > 1:
                break
            messages.append({"role": "user", "content": "Please give the final answer now."})

        # Budget exhausted (or repeated empty replies): answer from gathered evidence.
        st.trace.append(TraceStep(step=len(st.trace) + 1, kind="forced"))
        resp = await self._llm.complete(
            [*messages, {"role": "user", "content": FORCE_ANSWER}],
            tools=self._tool_specs,
            tool_choice="none",
            model=model,
            user=user,
        )
        st.account(resp)
        return clean_answer(resp.content or "") or FALLBACK_ANSWER

    @staticmethod
    def _check_operands(call: ToolCall, out: ToolOutcome, st: _RunState) -> ToolOutcome:
        try:
            exprs = [str(e) for e in json.loads(call.arguments).get("expressions", {}).values()]
        except (json.JSONDecodeError, AttributeError):
            return out
        bad = unsupported_operands(exprs, st.evidence)
        if not bad:
            return out
        return out.model_copy(
            update={
                "content": out.content
                + (
                    f"\nWARNING: these inputs do not appear in any earlier tool result: {bad}. "
                    "Re-check that you copied them exactly from the data, then recalculate."
                )
            }
        )

    async def _run_tools(
        self,
        resp: LLMResponse,
        messages: list[Message],
        st: _RunState,
        emit: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> None:
        calls = resp.tool_calls
        messages.append(
            {
                "role": "assistant",
                "content": resp.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                        **c.extra,
                    }
                    for c in calls
                ],
            }
        )

        async def run_one(i: int, call: ToolCall) -> tuple[ToolOutcome, int]:
            t0 = time.monotonic()
            sig = _call_signature(call)
            if i >= MAX_PARALLEL_CALLS:
                out = ToolOutcome(error=True, content="Too many tool calls in one turn; skipped.")
            elif sig in st.seen_calls:
                out = ToolOutcome(
                    error=True,
                    content=(
                        "You already made this exact call; its result is earlier in the "
                        "conversation. Use it, or change the arguments."
                    ),
                )
            else:
                st.seen_calls[sig] = call.id
                out = await self._tools.execute(call)
                if call.name == "calculate" and not out.error:
                    out = self._check_operands(call, out, st)
            return out, int((time.monotonic() - t0) * 1000)

        for call in calls:
            await emit("tool_call", {"name": call.name, "arguments": call.arguments})
        results = await asyncio.gather(*(run_one(i, c) for i, c in enumerate(calls)))

        for call, (out, ms) in zip(calls, results, strict=True):
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out.content})
            try:
                args: Any = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                args = call.arguments
            st.trace.append(
                TraceStep(
                    step=len(st.trace) + 1,
                    kind="tool",
                    name=call.name,
                    args=args,
                    output=_preview(out.content),
                    error=out.error,
                    latency_ms=ms,
                )
            )
            if not out.error:
                st.evidence.append(out.content)
                if call.name not in st.tools_used:
                    st.tools_used.append(call.name)
                for s in out.sources:
                    if s not in st.sources:
                        st.sources.append(s)
            st.degraded = st.degraded or out.degraded
            await emit(
                "tool_result",
                {"name": call.name, "error": out.error, "sources": [s.label for s in out.sources]},
            )
