"""Judge-vs-human agreement: Cohen's kappa and the labeling sample."""

from __future__ import annotations

from eval.judge_agreement import draw, kappa


def test_kappa():
    assert kappa([(True, True), (False, False)]) == 1.0
    assert kappa([(True, False), (False, True)]) == -1.0
    # 90% raw agreement, but the judge says "yes" 90% of the time and the human 80%: chance
    # alone would agree 74% of the time, so kappa = (0.90 - 0.74) / (1 - 0.74).
    pairs = [(True, True)] * 8 + [(True, False), (False, False)]
    assert round(kappa(pairs) or 0, 3) == 0.615
    assert kappa([(True, True)] * 3) is None  # no variation: kappa is undefined


def test_sample_keeps_judge_failures_both_sources_and_distinct_questions():
    # Two answers to each of 50 questions; the judge failed both answers to 10 of them.
    pool = [
        {
            "id": str(i),
            "source": "A" if i % 2 else "B",
            "question": f"q{i % 50}",
            "judge_passed": i % 5 != 0,
        }
        for i in range(100)
    ]
    sample = draw(pool, 30)
    assert len(sample) == 30 and len({x["question"] for x in sample}) == 30
    assert sum(not x["judge_passed"] for x in sample) == 10
    assert {x["source"] for x in sample} == {"A", "B"}
    assert all(x["human"] is None for x in sample)
    assert len(draw(pool, 60)) == 50  # one answer per question at most
