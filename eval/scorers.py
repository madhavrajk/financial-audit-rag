"""Scorers, one per `evaluation` type in the answer key, plus routing and refusal.

Design choices:
  * fuzzy_numeric: pass if ANY number in the answer (unit-normalised: "$416.2 billion" →
    4.162e11) is within 1% relative of the gold value (or ±0.05 for small gold values such
    as ratios and percentages). Lenient on formatting, strict on value.
  * exact_match_entity: the winning entity must be the first company named in the answer
    (the answer must lead with it), AND the number must match within 2% (or ±0.15).
  * llm_judge: a different model family from the agent judges against the gold answer with
    a fixed rubric (0 wrong / 1 partially correct / 2 correct), with temperature 0 and JSON
    output. Using a different family reduces self-preference bias (Zheng et al., 2023).
  * refusal: must say the data is unavailable and must not invent a figure for it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from app.core.errors import AppError
from app.core.grounding import extract_claims
from app.core.ports import LLMClient

COMPANIES = {
    "Apple": ("apple", "aapl"),
    "Microsoft": ("microsoft", "msft"),
    "Alphabet": ("alphabet", "google", "googl"),
}
REFUSAL = re.compile(
    r"not (?:in|included in|available in|covered by|part of) the (?:provided|available) "
    r"(?:data|dataset|sources)|(?:is|are) not available|not (?:provided|available)|"
    r"(?:no|does not have|doesn't have) (?:data|information)|only covers?|"
    r"only (?:includes|contains)|outside (?:the|of the) (?:provided|available)|"
    r"cannot (?:be )?(?:answer|determine)",
    re.I,
)


class Score(BaseModel):
    passed: bool
    score: float  # 0..1
    detail: str = ""
    judge_error: bool = False  # the judge call failed; excluded from accuracy, reported separately
    judge_cost_usd: float = 0.0  # what grading itself cost (so eval spend reconciles with billing)


def _numbers(answer: str) -> list[float]:
    return [v for _, v, _ in extract_claims(answer)]


def _num_match(values: list[float], gold: float, rel: float, abs_small: float) -> bool:
    for v in values:
        for g in (gold, gold * 1e9, gold * 1e6) if abs(gold) < 1e6 else (gold,):
            tol = abs_small if abs(g) < 100 else rel * abs(g)
            if abs(abs(v) - abs(g)) <= tol:
                return True
    return False


def fuzzy_numeric(answer: str, gold_numeric: float) -> Score:
    ok = _num_match(_numbers(answer), gold_numeric, rel=0.01, abs_small=0.051)
    return Score(passed=ok, score=float(ok), detail=f"gold={gold_numeric}")


def first_company(text: str) -> str | None:
    lowered = text.lower()
    best: tuple[int, str] | None = None
    for name, aliases in COMPANIES.items():
        for a in aliases:
            m = re.search(rf"\b{a}\b", lowered)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), name)
    return best[1] if best else None


def exact_match_entity(answer: str, gold_answer: str, gold_numeric: float | None) -> Score:
    want, got = first_company(gold_answer), first_company(answer)
    entity_ok = want is not None and want == got
    num_ok = gold_numeric is None or _num_match(_numbers(answer), gold_numeric, 0.02, 0.151)
    ok = entity_ok and num_ok
    return Score(
        passed=ok,
        score=(entity_ok + num_ok) / 2,
        detail=f"entity {got} vs {want} ({'ok' if entity_ok else 'MISS'}); "
        f"number {'ok' if num_ok else 'MISS'}",
    )


def refusal(answer: str) -> Score:
    said_unavailable = bool(REFUSAL.search(answer))
    return Score(
        passed=said_unavailable,
        score=float(said_unavailable),
        detail="declined correctly" if said_unavailable else "did not decline",
    )


JUDGE_SYSTEM = """\
You are grading an AI analyst's answer against a reference answer for a question about \
company 10-K filings. Judge factual agreement only, not style or length.

Score:
2 = correct: every key fact and number in the reference is present (rounding differences \
are fine, e.g. $416.2B vs $416.161B, 51.3% vs 51.31%) and nothing contradicts the reference.
1 = partially correct: some key facts are right but at least one is missing or wrong, or \
there is a material contradiction alongside correct content.
0 = incorrect: the main answer is wrong, missing, or it declines a question the reference \
answers.

Extra correct detail beyond the reference is fine. For descriptive lists (e.g. risk \
factors), require the main themes, not exact wording.

Respond with JSON only: {"score": 0|1|2, "missing": [..], "errors": [..], "rationale": "..."}"""


async def llm_judge(judge: LLMClient, model: str, question: str, gold: str, answer: str) -> Score:
    prompt = (
        f"Question:\n{question}\n\nReference answer:\n{gold}\n\n"
        f"Candidate answer:\n{answer}\n\nReturn the JSON verdict."
    )
    verdict: dict[str, Any] | None = None
    raw = ""
    judge_cost = 0.0
    for _ in range(3):  # reasoning judges occasionally return empty content; retry
        try:
            resp = await judge.complete(
                [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                model=model,
                max_tokens=4096,
            )
        except AppError as exc:
            raw = f"judge call failed: {exc}"
            continue
        judge_cost += resp.cost_usd
        raw = resp.content or ""
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            parsed = json.loads(m.group() if m else raw)
            int(parsed["score"])
            verdict = parsed
            break
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            continue
    if verdict is None:
        return Score(
            passed=False,
            score=0.0,
            judge_error=True,
            judge_cost_usd=judge_cost,
            detail=f"JUDGE_ERROR (excluded from accuracy): {raw[:150]}",
        )
    s = max(0, min(2, int(verdict["score"])))
    detail = verdict.get("rationale", "")
    if verdict.get("missing"):
        detail += f" | missing: {verdict['missing']}"
    if verdict.get("errors"):
        detail += f" | errors: {verdict['errors']}"
    return Score(passed=s == 2, score=s / 2, detail=detail[:600], judge_cost_usd=judge_cost)


TOOL_MODALITY = {"query_database": "sql", "search_filings": "pdf", "read_filing_pages": "pdf"}


def routing(tools_used: list[str], required: list[str]) -> dict[str, Any]:
    used = {TOOL_MODALITY[t] for t in tools_used if t in TOOL_MODALITY}
    return {
        "used": sorted(used),
        "required": sorted(required),
        "ok": set(required) <= used,
        "extra": sorted(used - set(required)),
    }
