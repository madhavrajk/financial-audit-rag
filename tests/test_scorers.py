from __future__ import annotations

import pytest

from eval import scorers


@pytest.mark.parametrize(
    "answer,gold,ok",
    [
        ("Apple's net sales in FY2025 were $416.2 billion.", 416161000000, True),
        ("Apple's revenue was $416,161 million.", 416161000000, True),
        ("Apple's revenue was $391.0 billion.", 416161000000, False),
        ("Apple's U.S. net sales were $151.8 billion.", 151790000000, True),
        ("Apple's Americas net sales were $178.4 billion.", 151790000000, False),
        ("Alphabet's diluted EPS was $10.81.", 10.81, True),
        ("Apple's current ratio was 0.89.", 0.89, True),
    ],
)
def test_fuzzy_numeric(answer, gold, ok):
    assert scorers.fuzzy_numeric(answer, gold).passed is ok


def test_exact_match_entity_requires_lead_entity_and_number():
    gold = "Alphabet, with net income up about 32.0%. Apple grew about 19.5%."
    good = "Alphabet grew net income the fastest, by 32.0% (Apple 19.5%, Microsoft 15.5%)."
    wrong_entity = "Apple grew net income the fastest at 19.5%; Alphabet grew 32.0%."
    wrong_number = "Alphabet grew net income the fastest, by 28.0%."
    assert scorers.exact_match_entity(good, gold, 32.0).passed
    assert not scorers.exact_match_entity(wrong_entity, gold, 32.0).passed
    r = scorers.exact_match_entity(wrong_number, gold, 32.0)
    assert not r.passed and r.score == 0.5


@pytest.mark.parametrize(
    "answer,ok",
    [
        ("Not in the provided data: the dataset covers only Apple, Microsoft and Alphabet.", True),
        ("Apple's FY2021 revenue is not available; data covers FY2023-FY2025.", True),
        ("Amazon's revenue was $638 billion.", False),
    ],
)
def test_refusal(answer, ok):
    assert scorers.refusal(answer).passed is ok


def test_routing():
    r = scorers.routing(["query_database", "calculate", "search_filings"], ["sql", "pdf"])
    assert r["ok"] and r["extra"] == []
    r = scorers.routing(["query_database"], ["sql", "pdf"])
    assert not r["ok"]


async def test_judge_failure_is_not_counted_as_wrong_answer():
    from tests.conftest import FakeLLM, say

    judge = FakeLLM([say(""), say("not json"), say("")])
    s = await scorers.llm_judge(judge, "m", "q", "gold", "answer")
    assert s.judge_error and not s.passed and len(judge.calls) == 3
    ok = await scorers.llm_judge(
        FakeLLM([say('{"score": 2, "rationale": "fine"}')]), "m", "q", "g", "a"
    )
    assert ok.passed and not ok.judge_error


async def test_eval_rows_carry_judge_error_and_summary_excludes_them():
    from eval.run_eval import run_question, summarize
    from tests.conftest import FakeLLM, say

    class FakeService:
        async def ask(self, question, use_cache=True):
            from app.service import ChatResult, Versions

            return ChatResult(
                answer="x",
                request_id="r",
                standalone_question=question,
                usage={"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0},
                versions=Versions(prompt="p", index="i", model="m"),
            )

    import asyncio

    q = {
        "id": "a",
        "tier": 1,
        "question": "q",
        "gold_answer": "g",
        "evaluation": "llm_judge",
        "required_modalities": [],
    }
    row = await run_question(FakeService(), q, FakeLLM([say("")] * 3), "m", asyncio.Semaphore(1))
    assert row["judge_error"] is True
    s = summarize([row])
    assert s["judge_errors_excluded"] == 1 and s["n"] == 0
